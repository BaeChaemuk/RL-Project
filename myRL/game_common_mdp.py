import pygame
import random
import time
from datetime import datetime
import numpy as np 
from tqdm import tqdm
import torch.nn as nn
import torch
from torch.distributions.categorical import Categorical
import torch.optim as optim
import pyautogui

class obj:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.move = 0
        
    def put_img(self, address):
        if address[-3:] == "png":
            self.img = pygame.image.load(address).convert_alpha()
        else :
            self.img = pygame.image.load(address)
        self.sx, self.sy = self.img.get_size()
    def change_size(self, sx, sy):
        self.img = pygame.transform.scale(self.img, (sx, sy))
        self.sx, self.sy = self.img.get_size()
    def show(self):
        screen.blit(self.img, (self.x,self.y))

class GAME:
    def __init__(self):
        self.map = np.zeros((1, 500, 900))
    def reset(self):

        self.spaceShip = obj()
        self.spaceShip.put_img("./image/spaceship.png")
        self.spaceShip.change_size(40, 80)
        self.spaceShip.x = round(self.map.shape[1]/2- self.spaceShip.sx/2)
        self.spaceShip.y = self.map.shape[2] - self.spaceShip.sy - 15
        self.spaceShip.move = 5

        self.meteorite_list = []
        self.missile_list = []

        self.k =0
        self.loss = 0
        self.kill = 0

        self.left_go = False
        self.right_go = False
        self.space_go = False

        self.begin_time = time.time()

    def mk_meteorite(self):
        if random.random() > 0.9:
            meteorite = obj()
            meteorite.put_img("./image/meteorite.png")
            meteorite_size = random.randrange(0,20)
            if meteorite_size == 19: 
                meteorite_size = 150
                meteorite_speed = 2
            elif meteorite_size < 19 and meteorite_size >= 15 : 
                meteorite_size = 100
                meteorite_speed = 3
            elif meteorite_size < 16 and meteorite_size >= 10 : 
                meteorite_size = 80
                meteorite_speed = 4
            else: 
                meteorite_size = 50
                meteorite_speed = 5
            meteorite.change_size(meteorite_size,meteorite_size) 
            meteorite.x = random.randrange(0, self.map.shape[1]-meteorite.sx-round(self.spaceShip.sx/2))
            meteorite.y = 10
            meteorite.move = meteorite_speed
            self.meteorite_list.append(meteorite)

    def fire(self):
        missiles = obj()
        missiles.put_img("./image/missile.png")
        missiles.change_size(5,15)
        missiles.x = round(self.spaceShip.x + self.spaceShip.sx/2 - missiles.sx/2)
        missiles.y = self.spaceShip.y - missiles.sy - 10
        missiles.move = 15
        self.missile_list.append(missiles)
    
    def missile_move(self):
        d_list = []    
        for i in range(len(self.missile_list)):
            missile = self.missile_list[i]
            missile.y -= missile.move
            if missile.y <= -missile.sy:
                d_list.append(i)
        d_list.reverse()
        for d in d_list:
            del self.missile_list[d]

    def meteorite_move(self):
        d_list = []    
        for i in range(len(self.meteorite_list)):
            meteorite_ = self.meteorite_list[i]
            meteorite_.y += meteorite_.move
            if meteorite_.y >= self.map.shape[2]:
                d_list.append(i)
        d_list.reverse()
        for d in d_list:
            del self.meteorite_list[d]
            loss += 1


    def strike(self):
        d_missile_list = []
        d_meteorite_list = []

        for i in range(len(self.missile_list)):
            for j in range(len(self.meteorite_list)):
                m = self.missile_list[i]
                a = self.meteorite_list[j]
                if self.crash(m,a) == True:
                    d_missile_list.append(i)
                    d_meteorite_list.append(j)
        d_missile_list = list(set(d_missile_list))
        d_meteorite_list = list(set(d_meteorite_list))
        d_missile_list.reverse()
        d_meteorite_list.reverse()
        try:
            for dm in d_missile_list:
                del self.missile_list[dm]
            for da in d_meteorite_list:
                del self.meteorite_list[da]
                self.kill += 1
                self.reward += 0.1
        except:
            pass

    def step(self, action):
       
        self.reward = 0
        now_time = datetime.now()
        elapsed_time = time.time() - self.begin_time
        self.delta_time = round((now_time - start_time).total_seconds())

        if elapsed_time >= time_limit:
            terminated = True
            pygame.quit()
        else : terminated = False
        
        if action == 1: # left go
            self.spaceShip.x -= self.spaceShip.move*3
            if self.spaceShip.x <= 0:
                self.spaceShip.x = 0
            self.reward += 0.001
        elif action == 2: # right go
            self.spaceShip.x += self.spaceShip.move*3
            if self.spaceShip.x >= self.map.shape[1] - self.spaceShip.sx:
                self.spaceShip.x = self.map.shape[1] - self.spaceShip.sx
            self.reward += 0.001
        elif action == 3 and self.k % 6 ==0: # fire
            self.fire()
            self.reward += 0.001
        elif action == 4 and self.k % 6 == 0: # left go and fire
            self.spaceShip.x -= self.spaceShip.move*3
            if self.spaceShip.x <= 0:
                self.spaceShip.x = 0
            self.fire()
            self.reward += 0.0011
        elif action == 5 and self.k % 6 == 0: # right go and fire
            self.spaceShip.x += self.spaceShip.move*3
            if self.spaceShip.x >= self.map.shape[1] - self.spaceShip.sx:
                self.spaceShip.x = self.map.shape[1] - self.spaceShip.sx
            self.fire()
            self.reward += 0.0011
        elif action == 0 : self.reward -= 0.001 # noob
        self.k+=1
        self.missile_move()

        self.mk_meteorite()
        self.meteorite_move()

        self.strike()
        for meteor in self.meteorite_list:
            if self.crash(meteor, self.spaceShip) == True : 
                self.reward -= 100
                terminated = True
                pygame.quit()
            else : 
                terminated = False
                self.reward +=0.01

        screen.fill(black)
        self.spaceShip.show()
        self.update_game_map(self.spaceShip, 1)
        for missile in self.missile_list:
            missile.show()
            self.update_game_map(missile,3)
        for meteorite in self.meteorite_list:
            meteorite.show()
            self.update_game_map(meteorite,2)
            
        truncated = False
        obs = self.get_map()


        font = pygame.font.Font("C:/Windows/Fonts/Arial.ttf", 20)
        text_kill = font.render("killed : {} loss : {}".format(self.kill, self.loss), True, (255,255,0))
        screen.blit(text_kill, (10, 5))
        
        text_time = font.render("time : {}".format(self.delta_time), True, (255,255,255))
        screen.blit(text_time, (size[0]-100, 5))

        if terminated ==  True: return obs, self.reward, terminated, truncated
        pygame.display.flip()

        return obs, self.reward, terminated, truncated
    def update_game_map(self, obj_, obj_id):
        obj = screen.blit(obj_.img, (obj_.x,obj_.y))
        start_x = obj[0]
        start_y = obj[1]
        end_x = obj[0] + obj[2]
        end_y = obj[1] + obj[3]
        self.map[start_x:end_x, start_y:end_y] = obj_id
    def get_map(self):
        return self.map  
    def crash(self,a, b):
        if (a.x-b.sx <= b.x) and (b.x <= a.x+a.sx):
            if (a.y-b.sy <= b.y) and (b.y <= a.y+a.sy):
                return True
            else:
                return False
        else : 
            return False

class Policy_Network(nn.Module):
    def __init__(self, obs_space_dims: int, action_space_dims: int):

        super().__init__()

        hidden_space1 = 16  # Nothing special with 16, feel free to change
        hidden_space2 = 32  # Nothing special with 32, feel free to change

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditioned on the observation, returns the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            x: Observation from the environment

        Returns:
            action_means: predicted mean of the normal distribution
            action_stddevs: predicted standard deviation of the normal distribution
        """
        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )

        return action_means, action_stddevs


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class RL(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(1, 16, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 1, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(6431, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, 6), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

        self.learning_rate = 1e-4  
        self.gamma = 0.99  
        self.eps = 1e-6 

        self.probs = [] 
        self.rewards = []  
        
        
    def get_value(self, obs):
        return self.critic(self.network(obs / 3.0))
    
    def get_action_value(self, obs):
        obs = torch.Tensor(np.array([obs]))
        hidden = self.network(obs / 3.0)
        logits = self.actor(hidden)
        prob = Categorical(logits= logits)
        self.probs.append(prob)
        action = prob.sample()
        
        return action, self.critic(hidden)
    
    def update(self):
        """Updates the policy network's weights."""
        running_g = 0
        gs = []

        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        log_probs = torch.stack(self.probs)

        log_prob_mean = log_probs.mean()

        loss = -torch.sum(log_prob_mean * deltas)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.probs = []
        self.rewards = []

    
#-----trian setup------
total_num_episodes = int(5e3)
learning_rate = 5e-5
epsilon = 5e-7
#-----trian setup------


#-----game setup------
time_limit = 30
black = (0,0,0)
white = (255,255,255)

size = [400, 900]

#-----game setup------


if __name__ == "__main__":
    

    rewards_over_seeds = []
    obs = torch.zeros(1,500,900)
    for seed in [1, 2, 3, 5, 8]:  # Fibonacci seeds
        # set seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        pygame.init()
        screen = pygame.display.set_mode(size)
        env = GAME()

        agent = RL()
        optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=epsilon)
        reward_over_episodes = []

        for episode in range(total_num_episodes):
            obs[0, 230:270, 805:885] = 1
            obs = torch.Tensor(obs)
            done = False
            start_time = datetime.now()
            env.reset()
            while not done:
                action,_ = agent.get_action_value(obs)
                obs, reward, terminated, truncated = env.step(action)
                agent.rewards.append(reward)
                done = terminated or truncated
            pygame.quit()
            reward_over_episodes.append(agent.rewards)
            agent.update()

            if episode % 1000 == 0:
                avg_reward = int(np.mean(reward_over_episodes))
                print("Episode:", episode, "Average Reward:", avg_reward)

        rewards_over_seeds.append(reward_over_episodes)