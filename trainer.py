"""General schema of training block.
Bases on some protocols, that should be implemented to run actual training:
- how to play game between players
- how to score results, and measure players improvements
"""
from typing import Any, Callable, List, Optional, Sequence
from datetime import datetime

# type aliases, for clarity of the further code
Player = Any
PlayerResultSummary = Any


# Protocol: Callable, that lets player play some games
# against an opponent and returns its results summary
TestGamePlayer = Callable[[Player], PlayerResultSummary]


# A training need to stop at some moment :)
# Protocol: Stopping condition - A callable, that is run to check,
# whether training process of a population of players should stop
StopCondition = Callable[[Sequence[Player]], bool]


# I define here stop condition, that waits till given metrics stop improving

# Protocol: Metrics of player results
Metrics = Callable[[PlayerResultSummary], float]


class NoImprovements:
    """stop condition, that waits till given metrics stop improving
    for the best player in the population"""

    def __init__(
        self,
        max_count: int,
        tests_player: TestGamePlayer,
        metrics: Metrics,
        patience: int,
    ):
        self.tests_player = tests_player
        self.metrics = metrics
        self.max_count = max_count
        self.patience = patience
        self.count = 0
        self.no_improvement_count = 0
        self.current_max_metrics: Optional[float] = None

    def __call__(self, players: Sequence[Player]):
        results = self.tests_player(players[0])  # the best one plays tests
        metrics = self.metrics(results)
        if self.current_max_metrics is None:
            # firs run only...
            self.current_max_metrics = metrics - 1

        self.count += 1

        if metrics > self.current_max_metrics:
            # metrics is improving
            self.no_improvement_count = 0
            self.current_max_metrics = metrics
        else:
            self.no_improvement_count += 1
        print(
            f"{datetime.now()}: Result: {metrics}, it is {self.no_improvement_count} steps without improvement."
        )

        # no improvements  or process takes too long
        return self.no_improvement_count > self.patience or self.count > self.max_count


#
# Training generally involves repeating the training steps until a stopping condition is met

# A callable, that trains a generation of players, and produces the next generation
# first players in the  next generation should be the best players from previous generation
TrainingStep = Callable[[Sequence[Player]], Sequence[Player]]

# I define here one training step: evolution, that
# plays test games and propagates best players

# the method to propagate depends on the players, here only a protocol
# Protocol: creates the next generation, based on ranking
Propagator = Callable[[Sequence[Player], List[int]], Sequence[Player]]


class EvolutionStep:
    def __init__(
        self,
        tests_player: TestGamePlayer,
        metrics: Optional[Metrics],
        propagator: Optional[Propagator],
    ):
        self.tests_player = tests_player
        self.metrics = metrics
        self.propagator = propagator

    def __call__(self, players: Sequence[Player]):
        results: List[PlayerResultSummary] = []  # the more the better
        for player in players:
            results.append(self.tests_player(player))
            print(f"Player result: {results[-1]}")

        # if there is no propagator or metrics, just return players
        if not self.propagator:
            return players
        if not self.metrics:
            return players

        metrics: List[float] = [-self.metrics(r) for r in results]
        # rank players and generate offspring
        ranking = sorted(range(len(players)), key=lambda i: metrics[i])
        # ranking[0] = 5 means, that 5th player was the best
        return self.propagator(
            players, ranking
        )  # this returns best player as the first


#
# main training procedure
def train(
    players: Sequence[Player],
    training_step: TrainingStep,
    stop_condition: StopCondition,
) -> Sequence[Player]:
    """runs training step, until stop condition is met"""
    while 1:
        players = training_step(players)
        if stop_condition(players):
            break
    return players
