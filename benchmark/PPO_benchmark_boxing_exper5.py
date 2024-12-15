import gymnasium as gym
from collections import defaultdict
import numpy as np 
from tqdm import tqdm
import torch.nn as nn
import torch
from torch.distributions.categorical import Categorical
import random
import torch.optim as optim
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
num_envs = 64
total_time_steps = 1500000
num_steps = 32
batch_size = 1024
minibatch_size = 128
num_iters = total_time_steps // batch_size
learning_rate = 2.5e-4
epsilon = 1e-5
discount_factor = 0.99
GAE_lamda = 0.95
policy_update_epochs = 4
clip_vloss = True
norm_adv = True
ent_coef =0.01
vf_coef = 0.5
max_grad_norm = 0.5
target_kl = None
seed_ = 214574
clip_coef_limit = 0.1
g_clip_coef = 0.8
num_minibatches = 4

def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda x: True)
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
    
class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(512, env.single_action_space.n), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(512, 1), std=1.0),
        )
    def get_value(self, obs):
        return self.critic(self.network(obs / 255.0))
    def get_action_value(self, obs, action=None):
        hidden = self.network(obs/255.0)
        logits_ = self.actor(hidden)
        distrib = Categorical(logits=logits_)

        if action is None:
            action = distrib.sample()
        return action, distrib.log_prob(action), distrib.entropy(), self.critic(hidden)
    
if __name__ == '__main__':
    
    minibatch_size = int(batch_size // num_minibatches)
    run_name = f"BoxingNoFrameskip-v0__ppo__improvement1_adaptiveClipping{seed_}__{int(time.time())}"

    
    torch.manual_seed(seed_)
    random.seed(seed_)
    np.random.seed(seed_)

    device = torch.device("cuda")

    envs = gym.vector.SyncVectorEnv(
        [make_env("BoxingNoFrameskip-v4", i, True, run_name) for i in range(num_envs)],
    )

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=epsilon)

    obs = torch.zeros((num_steps, num_envs) + (envs.single_observation_space.shape)).to(device)
    actions = torch.zeros((num_steps, num_envs) + (envs.single_action_space.shape)).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=seed_)
    next_obs = torch.Tensor(next_obs).to(device)
   # next_obs = next_obs.permute(2,0,1).to(device)
    next_done = torch.zeros(1).to(device).to(device)

    for iter in tqdm(range(1, num_iters)):
        frac = 1.0 - (iter - 1.0) / num_iters
        lr_ = frac * learning_rate
        optimizer.param_groups[0]['lr'] = lr_

        for step in range(0, num_steps):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + discount_factor * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + discount_factor * GAE_lamda * nextnonterminal * lastgaelam
            returns = advantages + values
        
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(policy_update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                clip_coef = 0.8
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                check = False
                
                for idx in range(len(pg_loss2)):
                    if pg_loss1[idx] < pg_loss2[idx]: check = True
                
                if check : clip_coef -= (g_clip_coef / ((batch_size // minibatch_size) +1))
                if clip_coef < clip_coef_limit : clip_coef = clip_coef_limit

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

            if target_kl is not None and approx_kl > target_kl:
                break
        
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        print("SPS:", int(global_step / (time.time() - start_time)))


    envs.close()