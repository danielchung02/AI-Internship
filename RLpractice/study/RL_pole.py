import numpy as np
import gymnasium as gym

env = gym.make("CartPole-v1")

state, info = env.reset()
done = False

while not done:
    action = np.random.choice([0, 1])

    next_state, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

    print("action:", action)

    state = next_state

env.close()