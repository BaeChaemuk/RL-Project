# RL-Project

benchmark 폴더는 bechmark환경에서 진행한 코드입니다.
myRL 폴더는 제가 직접 만든 환경에서 mdp를 정의한 부분입니다.

각 폴더에는 requiremets.txt파일이 있습니다.

benchmark 폴더에는 video폴더에 각각 학습된 게임 영상이 저장되어있으며 코드 실행시 1~26 episode 플레이 영상이 생성됩니다. 

benchmark/video 파일
BoxingNoFrameskip-v0__ppo__214574__1733829381 : 기본 PPO알고리즘으로 최대한 학습시킨 에이전트
BoxingNoFrameskip-v0__ppo__base214574__1733926586 : 대조군으로 사용하는 에이전트로 defalut PPO알고리즘에 학습함
BoxingNoFrameskip-v0__ppo__dicountFactor1__214574__1733945395 : discount factor를 1로 하여 PPO알고리즘에 학습한 에이전트
BoxingNoFrameskip-v0__ppo_discountFactor0__214574__1733941080 : discount factor를 0로 하여 PPO알고리즘에 학습한 에이전트
BoxingNoFrameskip-v0__ppo__noClipping214574__1733949696 : clipping 없이 PPO 알고리즘으로 학습한 에이전트

아래 실험은 적은 에피소드에서도 특별히 학습이 잘 되어서 seed만 바꾸어서 여러번 진행하였습니다.
BoxingNoFrameskip-v0__ppo__improvement1_adaptiveClipping_1_2__1734241325 : adaptive clipping을 사용하여 학습한 에이전트1
BoxingNoFrameskip-v0__ppo__improvement1_adaptiveClipping_1_5739__1734226787 : adaptive clipping을 사용하여 학습한 에이전트2
BoxingNoFrameskip-v0__ppo__improvement1_adaptiveClipping_1_214574__1733961502 : adaptive clipping을 사용하여 학습한 에이전트3
BoxingNoFrameskip-v0__ppo__improvement1_adaptiveClipping_1_812749__1734221481 : adaptive clipping을 사용하여 학습한 에이전트4
BoxingNoFrameskip-v0__ppo__improvement1_adaptiveClipping_1_21314125__1734216169 : adaptive clipping을 사용하여 학습한 에이전트5
BoxingNoFrameskip-v0__ppo__improvement1_adaptiveClipping_1_65798__1734232100 : adaptive clipping을 사용하여 학습한 에이전트6
