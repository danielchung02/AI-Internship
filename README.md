# AI-Internship

# AI Internship: Deep Reinforcement Learning

Gymnasium 환경에서 actor-critic 기반 강화학습 알고리즘을 직접 구현한 학습 코드입니다. 학습 코드의 구조를 가능한 한 통일해, 알고리즘별 차이를 비교하며 공부할 수 있도록 구성했습니다.

## Implemented algorithms

| Environment | Algorithm | Training file |
| --- | --- | --- |
| Pendulum-v1 | DDPG | `pendulum_ddpg.py` |
| HalfCheetah-v5 | DDPG | `halfcheetah_ddpg.py` |
| HalfCheetah-v5 | PPO | `halfcheetah_ppo.py` |
| HalfCheetah-v5 | TD3 | `halfcheetah_td3.py` |

## DDPG and TD3 structure

The DDPG implementations use an actor, critic, replay buffer, target networks, exploration noise, and soft target updates. `terminated` and `truncated` are stored separately: only true termination blocks bootstrapping, while time-limit truncation keeps the bootstrap term.

The TD3 implementation extends DDPG with:

- Twin critics (`Q1`, `Q2`) and clipped double-Q targets
- Target-policy smoothing with clipped target-action noise
- Delayed actor and target-network updates

## Training

Install the required packages first.

```bash
pip install torch gymnasium
pip install "gymnasium[mujoco]"
```

Run one of the training files.

```bash
python pendulum_ddpg.py
python halfcheetah_ddpg.py
python halfcheetah_ppo.py
python halfcheetah_td3.py
```

Each training script periodically evaluates the deterministic policy for 10 episodes. The latest checkpoint and the best-evaluation checkpoint are saved separately.

## Checkpoints and videos

Checkpoint files are saved under `pt/` and recorded videos are saved under `videos/`.

```text
pt/
├── pendulum_ddpg/
├── halfcheetah_ddpg/
├── halfcheetah_ppo/
└── halfcheetah_td3/

videos/
├── pendulum_ddpg/
├── halfcheetah_ddpg/
├── halfcheetah_ppo/
└── halfcheetah_td3/
```

To evaluate a saved model and record one episode, select an experiment at the top of `record.py` and run:

```python
EXPERIMENT = "halfcheetah_td3"
```

```bash
python record.py
```

By default, `record.py` loads each algorithm's latest checkpoint. Change `checkpoint_path` from `*_latest.pt` to `*_best.pt` to record the best evaluated model.
