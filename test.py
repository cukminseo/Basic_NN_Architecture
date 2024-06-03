import random

class Dice():
    def __init__(self, num_side, condition):
        self.num_side = num_side
        self.condition = condition

    def roll(self):
        score = random.randrange(self.num_side) + 1
        if not self.condition(score):
            raise 1
        return score

even_dice = Dice(6, lambda x: x % 2 == 0)
try:
    total_score = 0
    for _ in range(3):
        total_score += even_dice.roll()
        print('Win!')
except:
    print('Lose!')
finally:
    print('Total:', total_score)
