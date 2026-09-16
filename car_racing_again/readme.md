wrapper -> encoder -> agent 순서
근데 FrameStackObservation은 이미 gym에 있어서 따로 파일은 안 만듦

1. vec + DQN
VectorObservation → MLPEncoder → DQN

2. img + DQN
FrameStackObservation → CNNEncoder → DQN

3. vec + PPO
VectorObservation → MLPEncoder → PPO

4. img + PPO
FrameStackObservation → CNNEncoder → PPO

replay buffer는 dqn.py에 같이 들어가있음
rollout buffer도 ppo.py에