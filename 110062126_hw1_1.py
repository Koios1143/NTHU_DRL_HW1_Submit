import numpy as np

class GridWorld:
    def __init__(self, discount_factor=0.9 , grid_len=4):
        self.discount_factor = discount_factor
        self.grid_len = grid_len
        self.grid_size = grid_len ** 2
        self.actions = {'right': 0, 'left': 1, 'top': 2, 'bottom': 3}
        self.V = np.zeros(self.grid_size)
        self.history = []

    # state: A number within [0, 15], represent the grid world state
    # action: A string represent the action to take
    def transition(self, state, action):
        i, j = state // self.grid_len, state % self.grid_len
        if action == 'right':
            j = min(self.grid_len - 1, j + 1)
        elif action == 'left':
            j = max(0, j - 1)
        elif action == 'top':
            i = max(0, i - 1)
        elif action == 'bottom':
            i = min(self.grid_len - 1, i + 1)
        return i * self.grid_len + j

    def get_reward(self, state):
        if state == self.grid_size - 1:
            return 0
        else:
            return -1

    def policy_evaluation(self, epsilon=1e-6):
        self.V = np.zeros(self.grid_size)
        while True:
            V_new = np.zeros_like(self.V)
            for s in range(self.grid_size):
                if s == 0 or s == self.grid_size - 1:
                    continue
                v_s = 0
                for action, _ in self.actions.items():
                    s_prime = self.transition(s, action)
                    v_s += 0.25 * (self.get_reward(s) + self.discount_factor * self.V[s_prime])
                V_new[s] = v_s
            if np.max(np.abs(V_new - self.V)) < epsilon:
                break
            self.V = V_new
            self.history.append(self.V.copy())
        return self.V
    
if __name__ == "__main__":
    gammas = [1, 0.9, 0.1]
    for gamma in gammas:
        grid_world = GridWorld(discount_factor=gamma)
        state_values = grid_world.policy_evaluation()
        print(f'========== Gamma = {gamma} ==========')
        print(state_values)
        np.savetxt(f'110062126_hw1_1_data_gamma_{gamma}', state_values[1:-1], fmt='%.2f', delimiter=' ', newline=' ')
