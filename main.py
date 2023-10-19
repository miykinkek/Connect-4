from enum import Enum
from random import choice
from typing import Union

import numpy as np
from scipy.signal import convolve2d


class BoardColumnBusyError(Exception):
    ...


class BoardElement(Enum):
    def __str__(self):
        return self.value

    def __bool__(self):
        return self is not BoardElement.Empty

    Empty = "  "
    Red = "🔴"
    Yellow = "🟡"


class Board:
    def __init__(self):
        self._board = np.array([[BoardElement.Empty] * 7] * 6, dtype=BoardElement)

    def insert(self, idx: int, p: BoardElement):
        for row_idx, row in enumerate(self._board):
                if row[idx] is BoardElement.Empty:
                    self._board[row_idx][idx] = p
                    return
        raise BoardColumnBusyError(f"Column by {idx + 1} is fully busy, can not put {p.value} there!")

    def is_full(self) -> bool:
        return self._board.all()

    @property
    def board(self):
        return self._board

    def print(self):
        # rows are flipped so that the "gravity" vector matches the reality
        for row in np.flip(self._board, [0]):
            print("|".join([str(element) for element in row]))


def pick_first() -> BoardElement:
    first = choice([BoardElement.Red, BoardElement.Yellow])
    return BoardElement(first)


def change_player(p: BoardElement) -> BoardElement:
    return BoardElement.Yellow if p is BoardElement.Red else BoardElement.Red


def get_detection_kernels():
    horizontal_kernel = np.array([[1] * 4])
    vertical_kernel = np.transpose(horizontal_kernel)
    diag1_kernel = np.eye(4, dtype=np.uint8)
    diag2_kernel = np.fliplr(diag1_kernel)
    return [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]


def check_winning_move(b: Board, player: BoardElement) -> bool:
    detection_kernels = get_detection_kernels()
    return any(
        np.any(convolve2d(b.board == player, kernel) == 4)
        for kernel in detection_kernels
    )

def get_field_num() -> int:
    num = 0
    while not num:
        resp = input("Pick a spot: ")
        try:
            num = int(resp)
        except ValueError:
            print("Should be a number, try again")
        else:
            if num not in range(1, 8):
                print("Should be a number in range from 1 to 7")
                num = 0
    return num - 1


def turn(p: BoardElement, board: Board) -> tuple[bool, Union[BoardElement, None]]:
    print(f"{p.value}'s turn")
    print("Current board is: ")
    board.print()
    while True:
        try:
            field_num = get_field_num()
            board.insert(field_num, p)
        except BoardColumnBusyError as e:
            print(e)
        else:
            break
    someone_won = check_winning_move(board, p)
    if someone_won:
        print(f"Winner is: {p.value}")
        board.print()
        return True, p
    else:
        if not board.is_full():
            return False, change_player(p)
        print("It's a tie")
        board.print()
        return True, None


def main():
    game_over = False
    player = pick_first()
    board = Board()
    while not game_over:
        game_over, player = turn(player, board)


if __name__ == "__main__":
    main()
