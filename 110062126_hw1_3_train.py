import copy
import random
import numpy as np
from tqdm import trange

num_states = 19683
num_actions = 9
actions = [(i//3, i%3) for i in range(9)]
O_num = -1
X_num = 1
empty_num = 0

def to_state(state_index):
    state = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(9):
        state[i//3][i%3] = state_index % 3 - 1
        state_index = state_index // 3
    return copy.deepcopy(state)

def to_state_index(state):
    ret = 0
    for i in range(9):
        ret += (state[i//3][i%3] + 1) * (3**i)
    return ret

class Env():
    def __init__(self):
        self.current_state_num = 0
        self.current_state = []
        self.current_player = -1

    # Judge player_num won the game or not
    def judge(self, state, player_num):
        for i in range(3):
            if sum(state[i]) == 3 * player_num:
                return True
            if sum(np.array(state)[:, i].tolist()) == 3 * player_num:
                return True
        if sum([state[i][i] for i in range(3)]) == 3 * player_num:
            return True
        if sum([state[i][2-i] for i in range(3)]) == 3 * player_num:
            return True
        return False
    
    # check whether the state is valid
    # - No multiple winner
    # - abs(# of O - # of X) <= 1
    def is_valid(self, state, next_player):
        if self.judge(state, O_num) and self.judge(state, X_num):
            return False
        if np.abs(np.sum(np.array(state))) > 1:
            return False
        if np.abs(next_player + np.sum(np.array(state))) > 1:
            return False
        return True
    
    # check whether the state is terminal state
    # This function assume the state itself is valid
    # - Any player won the game
    # - Tie
    def is_terminal_state(self, state):
        if self.judge(state, O_num) or self.judge(state, X_num):
            return True
        count_symbols = lambda symbol: sum(row.count(symbol) for row in state)
        O, X = count_symbols(O_num), count_symbols(X_num)
        if (O == 5 and X == 4) or (O == 4 and X == 5):
            return True
        return False
    
    # Reset, generate current state with given start_player
    # Return generated state index
    def reset(self, current_player):
        self.current_player = current_player
        while True:
            self.current_state_num = random.randint(0, num_states-1)
            self.current_state = copy.deepcopy(to_state(self.current_state_num))
            if self.is_valid(self.current_state, self.current_player) and not self.is_terminal_state(self.current_state):
                break
        return self.current_state_num

    # player take an action A
    # Return next_state_num, reward, terminated
    def step(self, A, agent):
        action = (A//3, A%3)
        # Not a valid action
        bad_actions = [(x, y) for x in range(3) for y in range(3) if self.current_state[x][y] != empty_num]
        if action in bad_actions:
            return self.current_state_num, -100, True
        # A valid action
        self.current_state[action[0]][action[1]] = self.current_player
        self.current_state_num = to_state_index(self.current_state)

        if self.judge(self.current_state, self.current_player):
            return self.current_state_num, 20, True
        elif self.is_terminal_state(self.current_state):
            return self.current_state_num, 0, True

        new_action = agent.choose_action(np.array([self.current_player*-1] + np.array(self.current_state).reshape(-1).tolist()))
        self.current_state[new_action//3][new_action%3] = self.current_player*-1
        self.current_state_num = to_state_index(self.current_state)

        next_state, reward, terminated = self.current_state_num, 0, self.is_terminal_state(self.current_state)
        
        if self.judge(self.current_state, self.current_player):
            reward = 20
        elif self.judge(self.current_state, self.current_player*-1):
            reward = -20
        elif terminated:
            reward = 0

        return next_state, reward, terminated

class Agent():
    def __init__(self):
        self.Q = np.random.rand(num_states, num_actions) * 20 #np.zeros((num_states, num_actions))
        self.epsilon = 0.2
        self.epsilon_decay = 5e-7
        self.alpha = 0.7
        self.gamma = 0.6
    
    def init_Q_table(self, env:Env):
        for state in range(num_states):
            if env.is_terminal_state(to_state(state)):
                for action in range(num_actions):
                    self.Q[state, action] = 0

    def load_policy(self):
        self.Q = np.load('110062126_hw1_3_data')
    
    def save_policy(self):
        with open('110062126_hw1_3_data', 'wb') as f:
            np.save(f, self.Q)

    # Sample random action
    def sample(self):
        A = actions[random.randint(0, num_actions-1)]
        return A[0]*3+A[1]
    
    # choose best action
    def choose_action(self, query):
        current_player = query[0]
        state = np.array(query[1:]).reshape((3, 3)).tolist()
        state_idx = to_state_index(state)

        if current_player == X_num:
            state = (np.array(state)*(-1)).tolist()
            state_idx = to_state_index(state)

        return np.argmax(self.Q[state_idx])

    def epsilon_greedy(self, current_player, S):
        if np.random.rand() < self.epsilon:
            # perfrom random action
            return self.sample()
        else:
            #perform best action
            return self.choose_action(np.array([current_player] + np.array(to_state(S)).reshape(-1).tolist()))
    
    def update_policy(self, current_player, S, A, S_prime, reward):
        # update Q-table
        if current_player == X_num:
            current_state = (np.array(to_state(S)) * -1).tolist()
            S = to_state_index(current_state)
        else:
            next_state = (np.array(to_state(S_prime)) * -1).tolist()
            S_prime = next_state
        self.Q[S, A] += self.alpha * (reward + self.gamma * np.max(self.Q[S_prime, :]) - self.Q[S, A])
        self.epsilon -= self.epsilon_decay

def cannot_win(S, player, env):
        state = to_state(S)
        target = player * -1
        tmp = copy.deepcopy(state)
        for i in range(9):
            if state[i//3][i%3] == empty_num:
                tmp[i//3][i%3] = target
                if env.judge(tmp, target):
                    return True
                tmp[i//3][i%3] = empty_num
        return False

def plot_state(s, f):
    state = to_state(s)
    for row in state:
        f.writelines(str(row)[1:-1].replace(' ', '').replace('-1', 'O').replace('1', 'X').replace('0', ' ').replace(',', ' | '))
        f.writelines('\n')
    f.writelines('\n')

def eval(env, agent, logfile='log.txt', iter=10000):
    rewards2, wins2 = [], []
    with open(logfile, 'w') as f:
        for _ in trange(iter):
            init_player = [O_num, X_num][random.randint(0, 1)]
            next_player = init_player
            # Initialize S
            S = env.reset(next_player)
            terminated = False

            if cannot_win(S, next_player, env):
                continue
            f.writelines('O\n' if init_player == O_num else 'X\n')
            plot_state(S, f)

            # Total return for recording
            total_reward, win = 0, 0

            while not (terminated):
                A = agent.choose_action(np.array([next_player] + np.array(to_state(S)).reshape(-1).tolist()))
                f.writelines(f"{'O' if next_player == O_num else 'X'} {(A//3, A%3)}\n")
                if next_player == O_num:
                    f.writelines(f'{agent.Q[S]}\n')
                else:
                    tmp = to_state_index((np.array(to_state(S)) * -1).tolist())
                    f.writelines(f'{agent.Q[tmp]}\n')
                # Take action A, observe R, S'
                S_prime, reward, terminated = env.step(A, agent)
                plot_state(S_prime, f)
                
                # update total_reward
                total_reward += reward
                if terminated and reward >= 0:
                    win += 1
                elif terminated:
                    f.writelines(f'Lose\n')
                
                # update S
                S = S_prime
            rewards2.append(total_reward)
            wins2.append(win)
    print(f'Wining Rate: {sum(wins2)/len(wins2)*100}')
    return rewards2, wins2

def can_win(state, player, agent):
    env2 = Env()
    env2.current_state = to_state(state)
    env2.current_state_num = state
    env2.current_player = player
    S = state
    terminated = False

    while not (terminated):
        A = agent.choose_action(np.array([player] + np.array(to_state(S)).reshape(-1).tolist()))
        S_prime, reward, terminated = env2.step(A, agent)
        # update total_reward
        if terminated and reward >= 0:
            return True
        S = S_prime
        
    return False

def training(env, agent, iters=1000000, finetune=False):
    for iter in trange(iters):
        init_player = [O_num, X_num][random.randint(0, 1)]
        player = init_player
        # Initialize S
        S = env.reset(player)
        terminated = False

        if finetune:
            while can_win(S, player, agent):
                S = env.reset(player)

        while not (terminated):
            # Choose A from S using epsilon greedy
            A = agent.epsilon_greedy(player, S)
            # Take action A, observe R, S'
            S_prime, reward, terminated = env.step(A, agent)
            # update Q-table
            agent.update_policy(player, S, A, S_prime, reward)
            # update S
            S = S_prime
        
        if iter % 1000 == 0 and iter != 0:
            agent.save_policy()

if __name__ == '__main__':
    env, agent = Env(), Agent()
    agent.init_Q_table(env)
    # Start Training Iteration
    training(env, agent)
    
    # # Evaluate training
    eval(env, agent)

    # Finetune states that failed
    agent.load_policy()
    agent.epsilon_decay = 0

    training(env, agent, iters=1000000, finetune=True)
    
    eval(env, agent, logfile='log2.txt')