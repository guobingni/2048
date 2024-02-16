import numpy as np


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
        self.grid = np.copy(grid)

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
