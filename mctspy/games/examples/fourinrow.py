import numpy as np
from mctspy.games.common import TwoPlayersAbstractGameState, AbstractGameAction
from scipy.signal import convolve2d

class FourInRowMove(AbstractGameAction):
    def __init__(self, x_coordinate, y_coordinate, value):
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate
        self.value = value
    def __eq__(self,other):
        return self.x_coordinate == other.x_coordinate and \
                    self.y_coordinate == other.y_coordinate and \
                    self.value == other.value
    def __repr__(self):
        return "x:{0} y:{1} v:{2}".format(
            self.x_coordinate,
            self.y_coordinate,
            self.value
        )


class FourInRowGameState(TwoPlayersAbstractGameState):

    x = 1
    o = -1
    horizontal_kernel = np.array([[ 1, 1, 1, 1]])
    vertical_kernel = np.transpose(horizontal_kernel)
    diag1_kernel = np.eye(4, dtype=np.uint8)
    diag2_kernel = np.fliplr(diag1_kernel)
    detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]

    def __init__(self, state, next_to_move=1):
        if len(state.shape) != 2 or state.shape[0] != state.shape[1]:
            raise ValueError("Only 2D square boards allowed")
        self.board = state
        self.board_size = state.shape[0]
        self.next_to_move = next_to_move

    
    def check(self):
        for kernel in self.detection_kernels:
            con = convolve2d(self.board, kernel, mode="valid")
            if (con == 4).any():
                return True,False
            if (con == -4).any():
                return False,True
        return False,False
    
    @property
    def game_result(self):

        xwin,owin = self.check()
        if xwin:
            return self.x
        if owin:
            return self.o

        if np.all(self.board != 0):
            return 0.
        
        # if not over - no result
        return None

    def is_game_over(self):
        return self.game_result is not None

    def is_move_legal(self, move):
        # check if correct player moves
        if move.value != self.next_to_move:
            return False

        # check if inside the board on x-axis
        x_in_range = (0 <= move.x_coordinate < self.board_size)
        if not x_in_range:
            return False

        # check if inside the board on y-axis
        y_in_range = (0 <= move.y_coordinate < self.board_size)
        if not y_in_range:
            return False

        # finally check if board field not occupied yet
        return self.board[move.x_coordinate, move.y_coordinate] == 0

    def move(self, move):
        if not self.is_move_legal(move):
            raise ValueError(
                "move {0} on board {1} is not legal". format(move, self.board)
            )
        new_board = np.copy(self.board)
        new_board[move.x_coordinate, move.y_coordinate] = move.value
        if self.next_to_move == FourInRowGameState.x:
            next_to_move = FourInRowGameState.o
        else:
            next_to_move = FourInRowGameState.x

        return FourInRowGameState(new_board, next_to_move)

    def get_legal_actions(self):
        rows = (self.board == 0).sum(axis = 0) - 1
        cols = np.where(rows>=0)
        return [
            FourInRowMove(coords[0], coords[1], self.next_to_move)
            for coords in list(zip(rows[cols],*cols)) 
        ]
