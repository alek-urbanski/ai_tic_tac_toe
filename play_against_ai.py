import ai
import engine


def main():
    player = ai.AIPlayer()
    player.load("r2")
    player.set_training_mode(training_mode=ai.TrainingMode.PLAY)

    opponent = engine.ManualPlayer()

    print(engine.play_games(100, (player, opponent)))


if __name__ == "__main__":
    main()
