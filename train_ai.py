"""Implements all protocols needed by trainer module
and runs the training"""
from typing import List, Optional, Sequence
from datetime import datetime
import numpy as np
import engine
import ai
import trainer


# propagator protocol implementation
class Propagator:
    """generates the next generation, current best player is first in the new list"""

    def __init__(self, spread: float):
        self.spread = spread
        self.rng = np.random.default_rng()

    def __call__(
        self, players: Sequence[ai.AIPlayer], ranking: List[int]
    ) -> Sequence[ai.AIPlayer]:
        """creates the next generation"""
        next_generation: List[ai.AIPlayer] = []
        copy_first: int = len(players) // 20  # copy first 5% players
        mutate_from: int = len(players) // 2  # additionally mutate half children
        for i in range(len(players)):
            if i < copy_first:
                # copy first players
                next_generation.append(players[ranking[i]])

            else:
                # add mutated children of random parents (exponential distribution)
                p1: int = int(min(1, self.rng.exponential(0.1)) * (len(players) - 1))
                p2: int = p1
                while p2 == p1:
                    p2 = int(min(1, self.rng.exponential(0.1)) * (len(players) - 1))
                child: ai.AIPlayer = players[ranking[p1]] + players[ranking[p2]]
                if i > mutate_from:
                    child *= self.spread
                next_generation.append(child)

        return next_generation


# TestGamePlayer protocol
class PlayAgainstRandomPlayer:
    """plays provided number of games against randomly playing opponent"""

    def __init__(
        self,
        number_of_games: int,
        training_mode: Optional[ai.TrainingMode] = None,
        desc: Optional[str] = None,
    ) -> None:
        self.number_of_games = number_of_games
        self.desc: str = desc if desc else ""
        self.training_mode = training_mode

    def __call__(self, player: ai.AIPlayer):
        training_partner = engine.RandomPlayerByRules()

        current_training_mode: Optional[ai.TrainingMode] = None
        if self.training_mode:
            # I am changing training mode, so I need to save the old one
            current_training_mode = player.get_training_mode()[0]
            player.set_training_mode(training_mode=self.training_mode)

        # play testing games, save results of first player only
        results = engine.play_games(
            self.number_of_games, (player, training_partner), self.desc
        )[0]

        if current_training_mode:
            player.set_training_mode(
                training_mode=current_training_mode,
            )

        return results


# Metrics (the more the better)
def valid_games_ratio(result: engine.PlayerResultSummary) -> float:
    """share of games, where player did not make any invalid move"""
    invalid_moves = sum(
        [
            result[0].get(engine.PlayerResult.TECH_LOSS, 0),
            result[1].get(engine.PlayerResult.TECH_LOSS, 0),
        ]
    )
    all_games = sum(result[0].values()) + sum(result[1].values())
    return (all_games - invalid_moves) / all_games


def won_games_ratio(result: engine.PlayerResultSummary) -> float:
    """share of games, where player did not make any invalid move"""
    won_games = sum(
        [
            result[0].get(engine.PlayerResult.WIN, 0),
            result[1].get(engine.PlayerResult.WIN, 0),
        ]
    )
    all_games = sum(result[0].values()) + sum(result[1].values())
    return (won_games) / all_games


def player_score(result: engine.PlayerResultSummary) -> float:
    """scoring players results, the more the better"""
    return sum(
        [
            -1000 * result[0].get(engine.PlayerResult.TECH_LOSS, 0),
            -1000 * result[1].get(engine.PlayerResult.TECH_LOSS, 0),
            -800 * result[0].get(engine.PlayerResult.LOSS, 0),
            -700 * result[1].get(engine.PlayerResult.LOSS, 0),
            +300 * result[0].get(engine.PlayerResult.WIN, 0),
            +400 * result[1].get(engine.PlayerResult.WIN, 0),
            +100 * result[0].get(engine.PlayerResult.DRAW, 0),
            +100 * result[1].get(engine.PlayerResult.DRAW, 0),
        ]
    )


def learn_player(
    player: ai.AIPlayer,
    training_mode: ai.TrainingMode,
    learning_rate: float,
    metrics: trainer.Metrics,
) -> ai.AIPlayer:
    """individual player learns until given metrics stops improving"""

    player.set_training_mode(
        training_mode=ai.TrainingMode.LEARN_TO_WIN, learning_rate=learning_rate
    )
    players = trainer.train(
        players=[player],
        training_step=trainer.EvolutionStep(
            tests_player=PlayAgainstRandomPlayer(10),
            metrics=metrics,
            propagator=lambda x, _: x,
        ),
        stop_condition=trainer.NoImprovements(
            max_count=100,
            patience=3,
            metrics=metrics,
            tests_player=PlayAgainstRandomPlayer(200, ai.TrainingMode.PLAY),
        ),
    )
    return players[0]


def main():
    population_size: int = 5  # probably 100_000 times less than needed :)
    print(f"{datetime.now()}: Creating population of {population_size} players")
    players: Sequence[ai.AIPlayer] = []
    for i in range(population_size):
        print(f"{datetime.now()}: Player {i} is learning the basic rules")
        new_player = ai.AIPlayer(training_mode=ai.TrainingMode.PLAY)
        # player learns a little with reinforced learning
        PlayAgainstRandomPlayer(1000, training_mode=ai.TrainingMode.LEARN_RULES)(new_player)
        players.append(new_player)

    print(f"{datetime.now()}: Population starts evolving to learn winning strategy.")
    # now the evolution
    players = trainer.train(
        players=players,
        training_step=trainer.EvolutionStep(
            tests_player=PlayAgainstRandomPlayer(200),
            metrics=player_score,
            propagator=Propagator(0.001),
        ),
        stop_condition=trainer.NoImprovements(
            max_count=200,
            patience=10,
            metrics=player_score,
            tests_player=PlayAgainstRandomPlayer(500, ai.TrainingMode.PLAY),
        ),
    )
    players[0].save("./saved_models/xx")


if __name__ == "__main__":
    main()
