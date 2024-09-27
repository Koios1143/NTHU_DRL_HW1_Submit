import copy
import numpy as np


class Agent():
    def __init__(self):
        self.O_num = -1
        self.X_num = 1
        self.empty_num = 0
        self.states = []

        def create_state(idx, board):
            if idx == 9:
                self.states.append(copy.deepcopy(board))
                return
            for i in [-1, 0, 1]:
                board[idx//3][idx%3] = i
                create_state(idx+1, board)
        create_state(0, [[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    def to_state(self, state_index):
        return copy.deepcopy(self.states[state_index])
    def to_state_index(self, state):
        return self.states.index(state)


    def load_policy(self):
        import pathlib
        p = pathlib.Path(__file__).parent.resolve()
        self.Q = np.load(str(p/'110062126_hw1_3_data'))
    
    # choose best action
    def choose_action(self, query):
        current_player = query[0]
        state = np.array(query[1:]).reshape((3, 3)).tolist()
        state_idx = self.to_state_index(state)

        if current_player == self.X_num:
            state = (np.array(state)*(-1)).tolist()
            state_idx = self.to_state_index(state)

        A = np.argmax(self.Q[state_idx])
        return [A%3, A//3]

if __name__ == '__main__':
    agent = Agent()
    agent.load_policy()
    with open('../hw1-3_sample_input', 'r') as f:
        for state in f:
            query = np.array([int(i) for i in state[:-1].split(' ')])
            action = agent.choose_action(query)
            print(f'{action[0]} {action[1]}')