import numpy as np
'''
class Bandit:
    def __init__(self, arms = 10):
        self.rates = np.random.rand(arms)

    def play(self,arm):
        rate = self.rates[arm]
        if rate>np.random.rand():
            return 1
        else:
            return 0
        
bandit = Bandit()
Qs = np.zeros(10)
ns = np.zeros(10)

for n in range(10):
    action = np.random.randint(0,10)
    reward = bandit.play(action)

    ns[action] += 1
    Qs[action] += (reward - Qs[action]) / ns[action]
    print(Qs)
    '''

class Agent:
    def __init__(self,epsilon, action_size = 10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)
agent = Agent(epsilon = 0.1)

for step in range(10):
    action = agent.get_action()
    reward = np.random.randint(0, 2)  # 임시로 reward를 0 또는 1로 랜덤 설정

    agent.update(action, reward)

    print(f"\nstep {step + 1}")
    print("action:", action)
    print("reward:", reward)
    print("Qs:", agent.Qs)
    print("ns:", agent.ns)