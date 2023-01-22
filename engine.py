from typing import Dict, List, Optional, Protocol, Tuple
import numpy as np
import enum


# Possible game results


class PlayerResult(enum.Enum):
    """Possible results of the tic_tac_to_game game"""

    # loss/win due to a technical mistake (move breaks game rules)
    TECH_LOSS = enum.auto()
    TECH_WIN = enum.auto()

    # game was played till the end
    DRAW = enum.auto()
    LOSS = enum.auto()
    WIN = enum.auto()


# Game engine


class TicTacToeGame:
    def __init__(self):
        # An empty 3x3 board represented as a single-dimensional list.
        self.board: List[int] = [0] * 9
        # Values to be used to indicate the player's moves on the board.
        # 0 means an empty field.
        self.player_values: List[int] = [-1, 1]
        self.symbols: Dict[int, str] = {-1: "x", 1: "o"}

        # The current status of the game
        self.next_to_play: int = 0  # players are 0 and 1
        self.moves_played: int = 0

        # Initial values for the fields used to track the game's outcome.
        self.has_ended: bool = False
        self._result: Optional[Tuple[PlayerResult, PlayerResult]] = None

    @property
    def result(self) -> Tuple[PlayerResult, PlayerResult]:
        """result of a finished game or an exception if the game has not yet ended.
        This property makes typechecking easier as it has a known type.
        """
        if self._result:
            return self._result
        raise ValueError

    def get_result_for_current_player(self) -> Optional[PlayerResult]:
        """Returns the outcome of the game before the current player's move.
        If the game has not yet ended, the outcome will be None. This can be used by players
        to determine if they have lost the game on the previous opponent's move."""
        if not self.has_ended:
            return None

        return self.result[self.next_to_play]  # type: ignore

    def get_board_for_current_player(self) -> List[int]:
        """Returns the board with the symbols used to indicate the players
        updated so that the player 0 is always the next to play."""

        # The value 0 represents an empty field on the board
        if self.next_to_play == 0:
            return self.board

        # 1 to play next, I need to update symbols used to indicate players
        mapper: Dict[int, int] = {
            0: 0,  # empty field
            1: -1,  # player to play next: -1
            -1: 1,  # the opponent
        }  # dict[player named by the game: player in new coding]
        return [mapper[field_value] for field_value in self.board]

    def __repr__(self) -> str:
        """prints the board"""

        str_repr: str = ""
        for i in range(9):
            str_repr += f" {self.symbols.get(self.board[i], i)} "
            if (i + 1) % 3 == 0:
                str_repr += "\n"
                if i < 8:
                    str_repr += "-----------\n"
            else:
                str_repr += "|"
        return str_repr

    def verify_move(self, field_no: int) -> bool:
        """verifies if the move complies with the rules of the game"""
        if field_no < 0 or field_no > 8:
            return False
        if self.board[field_no] != 0:
            return False
        return True

    def _record_valid_move(self, field_no: int):
        """records a move on the board"""
        self.board[field_no] = self.player_values[self.next_to_play]
        self.moves_played += 1

    def _switch_player(self) -> None:
        """switches the active player for the next move"""
        self.next_to_play = [1, 0][self.next_to_play]

    def _check_win(self, player: int) -> bool:
        """search for three together"""

        def get_field(row: int, column: int) -> int:
            """maps 3x3 coordinates to the
            corresponding position in the board array"""
            return self.board[row * 3 + column]

        player_value = self.player_values[player]
        # rows
        for row in range(3):
            if (
                (get_field(row, 0) == player_value)
                and (get_field(row, 1) == player_value)
                and (get_field(row, 2) == player_value)
            ):
                return True
        # columns
        for column in range(3):
            if (
                (get_field(0, column) == player_value)
                and (get_field(1, column) == player_value)
                and (get_field(2, column) == player_value)
            ):
                return True
        # diagonals
        if (
            (get_field(0, 0) == player_value)
            and (get_field(1, 1) == player_value)
            and (get_field(2, 2) == player_value)
        ):
            return True
        if (
            (get_field(0, 2) == player_value)
            and (get_field(1, 1) == player_value)
            and (get_field(2, 0) == player_value)
        ):
            return True

        return False

    def play_move(self, field_no: int) -> None:
        """plays the provided move. The player making the move has already been set
        in self.next_to_play, so no additional parameter is needed."""

        def ordered_result(
            result: Tuple[PlayerResult, PlayerResult]
        ) -> Tuple[PlayerResult, PlayerResult]:
            """converts a tuple representing the game outcome, with the current player
            in the first position, to one where the starting player is in the first position
            """
            if self.next_to_play == 0:
                return result
            return result[1], result[0]

        if not self.verify_move(field_no):
            # invalid move
            self.has_ended = True
            self._result = ordered_result(
                (PlayerResult.TECH_LOSS, PlayerResult.TECH_WIN)
            )
        else:
            # play the move
            self._record_valid_move(field_no)

            if self._check_win(self.next_to_play):
                self.has_ended = True
                self._result = ordered_result((PlayerResult.WIN, PlayerResult.LOSS))

            # it was the last possible move and neither player was able to win
            if self.moves_played == 9:
                self.has_ended = True
                self._result = (PlayerResult.DRAW, PlayerResult.DRAW)

        self._switch_player()


# Player


class Player(Protocol):
    """protocol for playing tic-tac-toe"""

    def analyze_and_play_move(self, game: TicTacToeGame) -> TicTacToeGame:
        """I expect to be able to request a move from a player,
        and that would be all I need."""
        ...


# simple players
class ManualPlayer:
    """User plays via terminal"""

    def analyze_and_play_move(self, game: TicTacToeGame) -> TicTacToeGame:
        """requests a move from the user"""
        print(game)
        if game.has_ended:
            print(game.result)
            return game

        move: str = input(
            f"Select move for {game.symbols[game.player_values[game.next_to_play]]}"
        )
        if move.isnumeric():
            game.play_move(int(move))
        else:
            game.play_move(-1)  # invalid move

        return game


class RandomPlayerByRules:
    """plays randomly, but knows the rules of the game"""

    def __init__(self) -> None:
        self.rng = np.random.default_rng()

    def analyze_and_play_move(self, game: TicTacToeGame) -> TicTacToeGame:
        """compute move and save the situation for further analysis"""
        if game.has_ended:
            return game

        # play random move, first empty field
        board = game.get_board_for_current_player()
        shuffle = self.rng.choice(9, 9, replace=False)  # type: ignore
        for i in range(9):
            if board[shuffle[i]] == 0:
                game.play_move(shuffle[i])
                return game

        return game


# functions to automatically play the game

# type alias for summary of player results
PlayerResultSummary = Tuple[
    Dict[PlayerResult, int],  # counters of results, when player started the game
    Dict[PlayerResult, int],  # and when player was the second one
]


def play_tic_tac_toe(players: Tuple[Player, Player]) -> TicTacToeGame:
    """automatically plays one tic tac toe game"""
    game = TicTacToeGame()
    current_player = 0
    while not game.has_ended:
        game = players[current_player].analyze_and_play_move(game)
        current_player = [1, 0][current_player]  # select next player

    # give the loosing player a chance to analyze
    game = players[current_player].analyze_and_play_move(game)

    return game


def play_games(
    games_to_play: int,
    players: Tuple[Player, Player],
    desc: str = "",
) -> Tuple[PlayerResultSummary, PlayerResultSummary]:
    """plays games_to_play games between players in different orders
    and reports the summarized results for each player"""

    result_summary: Tuple[PlayerResultSummary, PlayerResultSummary] = (
        ({}, {}),
        ({}, {}),
    )
    players_order: List[int] = [0, 1]  # order of players in the particular game

    for _ in range(games_to_play):
        game: TicTacToeGame = play_tic_tac_toe(
            (players[players_order[0]], players[players_order[1]])
        )
        # save results (increase proper counter, create if not found)
        for player_index in range(2):
            current_player_position: int = players_order.index(player_index)
            counter = result_summary[player_index][current_player_position].get(
                game.result[current_player_position], 0
            )
            result_summary[player_index][current_player_position][
                game.result[current_player_position]
            ] = (counter + 1)

        # change order for the next game
        players_order = [players_order[1], players_order[0]]

    return result_summary
