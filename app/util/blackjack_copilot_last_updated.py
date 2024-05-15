from typing import Tuple, Any
import cv2
import numpy as np
import random


def basic_strategy(player_hand_total, dealer_upcard):
    blackjack_table = {
        # Hard totals
        # not start with ace or count ace as 1
        (4, 2): ("Hit", None, None, None),
        (4, 3): ("Hit", None, None, None),
        (4, 4): ("Hit", None, None, None),
        (4, 5): ("Hit", None, None, None),
        (4, 6): ("Hit", None, None, None),
        (4, 7): ("Hit", None, None, None),
        (4, 8): ("Hit", None, None, None),
        (4, 9): ("Hit", None, None, None),
        (4, 10): ("Hit", None, None, None),
        (4, 11): ("Hit", None, None, None),

        # Soft totals
        #
        (14, 2): ("Hit", "Hit", None, None),
        (14, 3): ("Hit", "Hit", None, None),
        (14, 4): ("Hit", "Hit", None, None),
        (14, 5): ("Hit", "Hit", None, None),
        (14, 6): ("Hit", "Hit", None, None),
        (14, 7): ("Hit", "Hit", None, None),
        (14, 8): ("Hit", "Hit", None, None),
        (14, 9): ("Hit", "Hit", None, None),
        (14, 10): ("Hit", "Hit", None, None),
        (14, 11): ("Hit", "Hit", None, None),

        # Surrender
        (16, 9): (None, None, "Surrender", None),
        (16, 10): (None, None, "Surrender", None),
        (16, 11): (None, None, "Surrender", None),

        # Pair splitting
        (8, 2): (None, None, None, "Split"),
        (8, 3): (None, None, None, "Split"),
        (8, 4): (None, None, None, "Split"),
        (8, 5): (None, None, None, "Split"),
        (8, 6): (None, None, None, "Split"),
        (10, 10): (None, None, None, "Split"),
    }

    key = (player_hand_total, dealer_upcard)
    strategy = blackjack_table.get(key, ("Stand", "Stand", None, None))
    return strategy


def detect_objects():
    # Place for object detection using OpenCV
    return ["playing cards"]


def simulate_blackjack():
    player_hand = int(input())
    dealer_upcard = int(input())

    print("Player's hand total:", player_hand)
    print("Dealer's upcard:", dealer_upcard)

    strategy = basic_strategy(player_hand, dealer_upcard)
    print("Strategy:", strategy)


def main():
    simulate_blackjack()


if __name__ == "__main__":
    main()