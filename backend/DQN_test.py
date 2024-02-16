import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import pathlib
import numpy as np
# from game2048_Env import Env2048

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    '''
    DQN model class.
        - contains convolution layers and
          fully-connected layers.
        - returns Q-values for each one
          of the 4 actions (L, U, R, D)
        - both policy and target network
          are instances of this class.
    '''

    def __init__(self):
        super(DQN, self).__init__()

        # first layer conv. layers, recieve one-hot
        # encoded (16, 4, 4) array as input
        self.conv1 = nn.Conv2d(16, 128, kernel_size=(1, 2))
        self.conv2 = nn.Conv2d(16, 128, kernel_size=(2, 1))

        # second layer conv. layers, recieve first layer as input
        self.conv11 = nn.Conv2d(128, 128, kernel_size=(1, 2))
        self.conv12 = nn.Conv2d(128, 128, kernel_size=(2, 1))
        self.conv21 = nn.Conv2d(128, 128, kernel_size=(1, 2))
        self.conv22 = nn.Conv2d(128, 128, kernel_size=(2, 1))

        # flattened shape
        first_layer = 4 * 3 * 128 * 2
        second_layer = 2 * 4 * 128 * 2 + 3 * 3 * 128 * 2
        self.fc = nn.Sequential(
            nn.Linear(first_layer + second_layer, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        x = x.to(device)
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))

        x11 = F.relu(self.conv11(x1))
        x12 = F.relu(self.conv12(x1))
        x21 = F.relu(self.conv21(x2))
        x22 = F.relu(self.conv22(x2))

        # flatten and concat layers, input for linear layer
        s1 = x1.shape
        s2 = x1.shape

        s11 = x11.shape
        s12 = x12.shape
        s21 = x21.shape
        s22 = x22.shape

        x1 = x1.view(s1[0], s1[1] * s1[2] * s1[3])
        x2 = x2.view(s2[0], s2[1] * s2[2] * s2[3])

        x11 = x11.view(s11[0], s11[1] * s11[2] * s11[3])
        x12 = x12.view(s12[0], s12[1] * s12[2] * s12[3])
        x21 = x21.view(s21[0], s21[1] * s21[2] * s21[3])
        x22 = x22.view(s22[0], s22[1] * s22[2] * s22[3])

        concat = torch.cat((x1, x2, x11, x12, x21, x22), dim=1)
        return self.fc(concat)


class Env2048:
    '''
	fast and simple 2048 game environment.
		- inherits basic methods from Env
		- implements game logic
		- recommended for training, bad UI
	'''

    def __init__(self):
        self.grid = np.zeros((4, 4), dtype=int)
        self.add_tile()
        self.add_tile()
        self._score = 0

    def reset(self):
        return self.grid

    def add_tile(self):
        # add a new tile (2/4) in a random empty place
        i, j = (self.grid == 0).nonzero()
        rnd = np.random.choice(len(i))
        self.grid[i[rnd], j[rnd]] = np.random.choice([1, 2], p=[0.9, 0.1])

    def step(self, action):
        # rotate the board and step left
        rotated_board = np.rot90(self.grid, action)
        next_state = np.zeros((4, 4), dtype=int)

        # try to merge tiles in each column
        for col_idx in range(4):
            col = rotated_board[col_idx, :]
            result = np.zeros(4, dtype=col.dtype)
            j, prev = 0, None
            for i in range(4):
                if col[i] != 0:
                    # move tile to next empty place
                    if prev is None:
                        prev = col[i]
                    # merge tiles
                    elif prev == col[i]:
                        result[j] = col[i] + 1
                        self._score += 1 << result[j]
                        j += 1
                        prev = None
                    else:
                        result[j] = prev
                        j += 1
                        prev = col[i]
            if prev is not None:
                result[j] = prev
            next_state[col_idx] = result

        # rotate back and return to original state
        next_state = np.rot90(next_state, -action)
        if not (next_state == self.grid).all():
            self.grid = next_state
            self.add_tile()

        # returns the reward - #empty tiles
        return self.empty_tiles()

    def score(self):
        return self._score

    def state(self):
        return self.grid

    def set_board(self, grid):
        _grid = np.copy(grid)
        for i in range(4):
            for j in range(4):
                _grid[i][j] = np.log2(_grid[i][j]) if _grid[i][j] != 0 else 0
        self.grid = _grid

    def render(self):
        print('-' * 25)
        for i in range(4):
            for j in range(4):
                print((1 << self.grid[i][j] if self.grid[i][j] != 0 else "0"), end="\t")
            print()
        print('-' * 25)

    # returns if the game is over
    def is_done(self):
        if self.empty_tiles() > 0:
            return False
        grid = self.state()
        for i in range(4):
            for j in range(4):
                if i != 0 and grid[i - 1][j] == grid[i][j] or \
                        j != 0 and grid[i][j - 1] == grid[i][j]:
                    return False
        return True

    # maximal tile in the current board as a power of 2
    def max_tile(self):
        return 1 << self.state().max()

    def _can_perform(self, action):
        tmp = np.rot90(self.state(), action)
        for i in range(4):
            empty = False
            for j in range(4):
                empty |= tmp[i, j] == 0
                if tmp[i, j] != 0 and empty:
                    return True
                if j > 0 and tmp[i, j] != 0 and tmp[i, j] == tmp[i, j - 1]:
                    return True
        return False

    # returns a list of all possible actions
    def possible_actions(self):
        res = []
        for action in range(4):
            if self._can_perform(action):
                res.append(action)
        return res

    # amount of empty tiles in current board
    def empty_tiles(self):
        grid = self.state()
        count = 0
        for i in range(4):
            for j in range(4):
                count += (grid[i, j] == 0)
        return count


def eval_DQN(grid):
    moves = {0: 3, 1: 0, 2: 1, 3: 2}
    path = pathlib.Path('.').absolute() / 'backend' / 'Models' / 'target_net.pth'
    target_net = DQN().to(device)
    target_net.load_state_dict(torch.load(path, map_location=device))
    env = Env2048()
    env.set_board(grid)
    # print(grid)
    # print(env.possible_actions())
    state = torch.from_numpy(one_hot_encode(env.state())).to(device)
    actions = target_net(state).argsort()[0].cpu().numpy()[::-1]
    for action in actions:
        if action in env.possible_actions():
            return moves[action]
    raise ValueError()


def one_hot_encode(state):
    '''
    one-hot encode (4, 4) numpy array to (16, 4, 4) numpy array
        - each channel 0..15 is a (4, 4) numpy array,
        - conatins 1's where original grid contains 2^i
          (first channel refers for empty tiles)
    '''
    result = np.zeros((1, 16, 4, 4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            result[0][state[i][j]][i][j] = 1.0
    return result
