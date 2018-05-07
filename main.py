import numpy as np
import warnings


warnings.filterwarnings('error')


class Agent:

    def __init__(self, alpha, temp, gamma, reward, init_state=0):

        self.alpha = alpha
        self.temp = temp
        self.gamma = gamma

        self.reward = reward

        self.n_state = len(reward)
        self.q = np.zeros((self.n_state, self.n_state ))

        self.state = init_state

        self.action = None

    def choose(self):

        possible = self.reward[self.state] != - 1

        if sum(possible) > 1:
            p = self.softmax(self.q[self.state, possible], temp=self.temp)
            self.action = np.random.choice(np.arange(self.n_state)[possible], p=p)

        else:
            self.action = np.arange(self.n_state)[possible][0]

        return self.action

    @staticmethod
    def softmax(x, temp):
        try:
            return np.exp(x / temp) / np.sum(np.exp(x / temp))
        except Warning as w:
            print(x, temp)
            raise Exception(f'{w} [x={x}, temp={temp}]')

    def learn(self, new_state):

        self.q[self.state, self.action] += self.alpha * (
            self.reward[self.state, self.action] +
            self.gamma * np.max(self.q[new_state, self.reward[new_state] != - 1])
            - self.q[self.state, self.action]
        )

        self.state = new_state


def main():

    reward_matrix = np.array(
        [
            # 0  1   2   3  4   5
            [-1, -1, -1, -1,  0, -1],  # 0
            [-1, -1, -1,  0, -1,  1],  # 1
            [-1, -1, -1,  0, -1, -1],  # 2
            [-1,  0,  0, -1,  0, -1],  # 3
            [0,  -1, -1,  0, -1,  1],  # 4
            [-1,  0, -1, -1,  0,  1],  # 5
        ]
    )

    a = Agent(alpha=0.1, temp=0.05, gamma=0.8, reward=reward_matrix, init_state=2)

    for i in range(100):

        a.state = 2
        t = 0
        while True:

            act = a.choose()
            print(f'Agent chose {act}')
            new_state = act
            a.learn(new_state=new_state)

            t += 1

            if act in (5, ):
                print(f'Attempt {i}: Reach the end at t={t}!')
                print()
                break


if __name__ == '__main__':
    main()
