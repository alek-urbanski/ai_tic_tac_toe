from __future__ import annotations
from typing import List, Optional, Tuple
import enum
import numpy as np

import tensorflow as tf  # type: ignore

import engine


class TrainingMode(enum.Enum):
    """training modes for AIPlayer"""

    LEARN_RULES = enum.auto()
    LEARN_TO_WIN = enum.auto()
    PLAY = enum.auto()


class AIPlayer:
    """neural network that plays tic-tac-toe, with some learning options"""

    def __init__(self, training_mode: TrainingMode = TrainingMode.LEARN_RULES) -> None:
        self.layer_sizes = [
            36,
            36,
            24,
        ]  # based on some Internet articles, this should be more than enough

        # build model
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.Input(shape=(9)))  # type: ignore

        # dense layers from configuration parameter
        for i, layer_size in enumerate(self.layer_sizes):
            self.model.add(  # type: ignore
                tf.keras.layers.Dense(
                    layer_size,
                    activation="relu",
                    bias_initializer="glorot_normal",
                    name=f"D_{i}",
                )
            )
        # classification layers
        self.model.add(tf.keras.layers.Dense(9, name="classification_layer"))  # type: ignore
        self.model.add(tf.keras.layers.Softmax())  # type: ignore

        # initial training parameters
        self.learning_rate: float = 0.001
        self.model.compile(  # type: ignore
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        )

        self.training_mode: TrainingMode = training_mode

        # remember the last move, to analyze if opponent wins just after it
        self.last_board: List[int] = []
        self.last_move: int = -1

        # random number generator used for mating
        self.rng = np.random.default_rng()

    def set_training_mode(
        self,
        training_mode: Optional[TrainingMode] = None,
        learning_rate: Optional[float] = None,
    ):
        if training_mode:
            self.training_mode = training_mode
        if learning_rate is not None:
            self.learning_rate = learning_rate
            tf.keras.backend.set_value(self.model.optimizer.lr, self.learning_rate)  # type: ignore

    def get_training_mode(self) -> Tuple[TrainingMode, float]:
        return self.training_mode, self.learning_rate

    def _get_move(self, game: engine.TicTacToeGame) -> int:
        """computes the move from the model"""
        self.last_board = game.get_board_for_current_player()
        result: List[int] = self.model(  # compute move from the model
            tf.constant(self.last_board, shape=(1, 9))  # type: ignore
        ).numpy()[  # type: ignore
            0
        ]

        self.last_move = sorted(range(9), key=lambda i: result[i])[
            -1
        ]  # field no with max score
        return self.last_move

    def save(self, filename: str) -> None:
        self.model.save_weights(filename)  # type: ignore

    def load(self, filename: str) -> None:
        self.model.load_weights(filename)  # type: ignore

    def analyze_and_play_move(self, game: engine.TicTacToeGame) -> engine.TicTacToeGame:
        """draw conclusions from the situation  and play the next move"""
        # learn from the loss, player probably could block the opponent
        # if game.has_ended and self.training_mode == TrainingMode.LEARN_RULES:
        #    if game.get_result_for_current_player() == engine.PlayerResult.LOSS:
        #        self._discourage_last_move()
        #    return game

        # play move
        game.play_move(self._get_move(game))

        if self.training_mode in [TrainingMode.LEARN_RULES, TrainingMode.LEARN_TO_WIN]:
            # learn from the win or breaking the rules
            if game.has_ended:
                # current_player is already the next player, so the inversed results
                if game.get_result_for_current_player() == engine.PlayerResult.TECH_WIN:
                    self._discourage_last_move()
                if game.get_result_for_current_player() == engine.PlayerResult.LOSS:
                    self._endorse_last_move()
            elif self.training_mode == TrainingMode.LEARN_RULES:
                #  in this mode I am happy with just a valid move
                self._endorse_last_move()  # still in the game, the move wasn't an error

        return game

    def _discourage_last_move(self) -> None:
        """move was wrong, learn not to move like this any more"""
        # record last way of thinking
        with tf.GradientTape() as tape:
            result = self.model(  # type: ignore
                tf.constant(self.last_board, shape=(1, 9)), training=True  # type: ignore
            )[0]
            loss = result[self.last_move]  # type: ignore
        trainable_vars = self.model.trainable_variables  # type: ignore
        gradients = tape.gradient(loss, trainable_vars)  # type: ignore
        # Update weights
        self.model.optimizer.apply_gradients(zip(gradients, trainable_vars))  # type: ignore

    def _endorse_last_move(self) -> None:
        """move was good, learn to move like this"""
        # record last way of thinking
        with tf.GradientTape() as tape:
            result = self.model(  # type: ignore
                tf.constant(self.last_board, shape=(1, 9)), training=True  # type: ignore
            )[0]
            loss = 1 - result[self.last_move]  # type: ignore

        trainable_vars = self.model.trainable_variables  # type: ignore
        gradients = tape.gradient(loss, trainable_vars)  # type: ignore
        # Update weights
        self.model.optimizer.apply_gradients(zip(gradients, trainable_vars))  # type: ignore

    def __add__(self, partner: AIPlayer) -> AIPlayer:
        """mate with a partner, transfer each weight to the child randomly from one parent
        dunder method, just to have a fancy: child = partner1 + partner2 :)"""

        def random_join(a, b):  # type: ignore
            """returns array with elements randomly taken from a or b"""
            chooser = self.rng.integers(low=0, high=2, size=a.shape)  # type: ignore
            return a * chooser - b * (chooser - 1)  # type: ignore

        child_weights = []
        for (weights1, weights2) in zip(  # type: ignore
            self.model.get_weights(), partner.model.get_weights()  # type: ignore
        ):
            child_weights.append(random_join(weights1, weights2))  # type: ignore

        child = AIPlayer(training_mode=self.training_mode)
        child.model.set_weights(child_weights)  # type: ignore
        return child

    def __imul__(self, spread: float) -> AIPlayer:
        """mutate player, player *= spread,
        spread > 0.5 gives totally random player as a result"""
        random_player = AIPlayer()  # source of noise

        new_weights = []
        for (weights, noise) in zip(  # type: ignore
            self.model.get_weights(), random_player.model.get_weights()  # type: ignore
        ):
            new_weights.append(weights + spread * noise)  # type: ignore

        self.model.set_weights(new_weights)  # type: ignore
        return self
