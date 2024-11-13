from Gridworld import Gridworld


class GrdidWorldTask1:

    def __init__(self):
        self.game = Gridworld(size=4, mode='static')
        self.game.display()


if __name__ == "__main__":
    gridWorld = GrdidWorldTask1()
    print(gridWorld.game.display())
