import numpy as np
import copy
from tqdm import trange

player_O = -1
player_X = 1
empty = 0

class Board(object):
    def __init__(self, start_player):
        self.current_player = start_player
        self.availables = [i for i in range(64)]
        self.states = {}
    
    def load_board(self, filename):
        content = open(filename, 'r').read()
        self.start_player = int(content[0])
        board = content[:-1].split(' ')[1:]
        board = [int(i) for i in board]
        for i in range(64):
            if board[i] != empty:
                self.states[i] = board[i]

    def location_to_index(self, location):
        """
        Transform a (x, y, z) into an integer encode
        """
        x, y, z = location
        return x + 4*y + 16*z + 1
    
    def move_to_location(self, move):
        """
        Transform a integer encode into a (x, y, z)
        """
        move -= 1
        return move % 4, (move//4) % 4, (move//16) % 4

    def switch_player(self):
        self.current_player *= -1

    def do_move(self, move):
        """
        Given a encoded location (move), try to make a step
        """
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.switch_player()
    
    def has_winner(self):
        # Build current board in numpy 3D array
        board = np.zeros((4, 4, 4))
        for move, player in self.states.items():
            # move, player = state
            location = self.move_to_location(move)
            board[location] = player

        # Check whether any player won the game
        for player in [player_O, player_X]:
            for i in range(4):
                for j in range(4):
                    if sum(board[i, j, :]) == 4 * player \
                    or sum(board[i, :, j]) == 4 * player \
                    or sum(board[:, i, j]) == 4 * player:
                        return True, player
                if sum(np.diag(board[i])) == 4 * player \
                    or sum(np.diag(np.fliplr(board[i]))) == 4 * player:
                    return True, player
            if sum(board[i, i, i] for i in range(4)) == 4 * player \
                or sum(board[i, i, 3-i] for i in range(4)) == 4 * player \
                or sum(board[i, 3-i, i] for i in range(4)) == 4 * player \
                or sum(board[i, 3-i, 3-i] for i in range(4)) == 4 * player:
                return True, player
        return False, empty
    
    def is_terminate(self):
        """
        Check whether current state is terminal state
        Note that we consider the state is always legal state
        """
        win, winner = self.has_winner()
        if win:
            return True, winner
        if len(self.availables) <= 0:
            return True, empty
        return False, empty

    def get_current_player(self):
        return self.current_player

class Game(object):
    def __init__(self, board:Board):
        self.board = board

    def play(self, player1, player2, start_player):
        """
        We always regard player1 plays O
        While player2 plays X
        """
        while True:
            current_player = player1 if self.board.get_current_player() == player_O else player2
            move = current_player.choose_action(self.board)
            self.board.do_move(move)
            end, winner = self.board.is_terminate()
            if end:
                # print(f'Winner: {"O" if winner == player_O else "X"}')
                return winner

class TreeNode(object):
    def __init__(self, parent, c=5):
        self.parent = parent
        self.children = {} # action: TreeNode
        self.visit_count = 0
        self.total_reward = 0
        self.c = c
    
    def UCB(self):
        return (self.total_reward / (self.visit_count + 1)) + self.c * (np.sqrt(np.log(self.parent.visit_count + 1)) / (self.visit_count + 1))
    
    def select(self):
        """
        Select the best action according to UCB
        Return (action, TreeNode)
        """
        if self.is_leaf_node():
            return -1
        return max(self.children.items(), key=lambda child: child[1].UCB())

    def expand(self, actions):
        """
        Expand actions from current node
        """
        for action in actions:
            if action not in self.children:
                self.children[action] = TreeNode(self)

    def update(self, reward):
        self.total_reward += reward
    
    def backpropagate(self, reward):
        self.update(reward)
        if self.parent != None:
            self.parent.update(reward)

    def is_leaf_node(self):
        return self.children == {}

    def is_root_node(self):
        return self.parent == None
    
class MCTS(object):
    def __init__(self, c=5, playout_num=1000):
        self.root = TreeNode(None)
        self.playout_num = playout_num
    
    def play_once(self, state:Board):
        """
        Try to perform MCTS once, return reward
        """
        # Selection
        node = self.root
        while not node.is_leaf_node():
            action, node = node.select()
            state.do_move(action)
        # Expansion
        end, winner = state.is_terminate()
        if not end:
            node.expand(state.availables)
        # Roll-out
        reward = self.roll_out(state)
        # Backpropagate
        node.backpropagate(reward)
    
    def roll_out(self, state:Board):
        """
        Roll out, return reward
        """
        current_player = state.get_current_player()
        end, winner = state.is_terminate()
        while not end:
            action = state.availables[np.random.randint(0, len(state.availables))]
            state.do_move(action)
            end, winner = state.is_terminate()
        if current_player == winner:
            return 1
        elif winner == empty:
            return 0
        else:
            return -1
    
    def get_move(self, state):
        """
        Choose best move at given state
        """
        for _ in range(self.playout_num):
            state_copy = copy.deepcopy(state)
            self.play_once(state_copy)
        return max(self.root.children.items(), key=lambda child: child[1].visit_count)[0]
            
class Agent(object):
    def __init__(self, c=5, playout_num=1000, random=False):
        self.mcts = MCTS(c, playout_num)
        self.random = random
    
    # def choose_action(self, state:Board):
    #     if len(state.availables) <= 0:
    #         return -1
    #     if self.random:
    #         return state.availables[np.random.randint(0, len(state.availables))]
    #     else:
    #         move = self.mcts.get_move(state)
    #         self.mcts.root = TreeNode(None, 5)
    #         return move
        
    def choose_action(self, state:np.array):
        start_player = state[0]
        board = Board(start_player)

        for i in range(1, 65):
            if state[i] != empty:
                board.states[i] = state[i]

        move = self.mcts.get_move(board)
        self.mcts.root = TreeNode(None, 5)
        return [i for i in board.move_to_location(move)]
    
    def load_policy(self):
        pass

if __name__ == '__main__':
    state = np.array([int(i) for i in open('../hw1-4_sample_input', 'r').read()[:-1].split(' ')])
    agent = Agent()
    agent.load_policy()
    agent.choose_action(state)
