DEBUG_VIDEO = False

import os
import sys
import cv2

fabspath = os.path.abspath(__file__)
yoluster_dir = os.path.join(fabspath, "..", "..", "..", "ML")
sys.path.append(yoluster_dir)

from yoluster import YOLOCluster

# https://www.blackjackapprenticeship.com/blackjack-strategy-charts
# 0=S (Stand)
# 1=H (Hit)
# 2=D (Double if allowed, otherwise hit)
# 3=Ds (Double if allowed, otherwise stand)

HARDTOTALS = """0000000000
0000011111
0000011111
0000011111
0000011111
1100011111
2222222222
2222222211
1222211111
1111111111""".splitlines()

SOFTTOTALS = """0000000000
0000300000
3333300111
1222211111
1122211111
1122211111
1112211111
1112211111""".splitlines()

# 4 = Y (Split the pair)
# 5 = N (Don't split the pair)
# 6 = Y/N (Split only if 'DAS' is offered)

# DAS means Double After Split.

PAIRSPLITTING = """4444444444
5555555555
4444454455
4444444444
4444445555
6444455555
5555555555
5556655555
6644445555
6644445555""".splitlines()

# 7 = null
# 8 = SUR (Surrender)

LATESURRENDER = """7777777888
7777777787
7777777777""".splitlines()

PICTURECARD = "JKQ"


def get_sum_of_cards(hard_count: int, number_of_ace: int) -> int:
    if number_of_ace == 0:
        return hard_count

    if number_of_ace == 1:

        # Ace can be interpreted as 11 or 1, to make sum closer to 21 (but less than or equal to 21).

        if hard_count < 11:
            return hard_count + 11
        return hard_count + 1

    # only up to one ace card can be interpreted as 11 because 11*2=22 exceeds 21.
    ret = hard_count + number_of_ace - 1

    if ret < 11:
        return ret + 11
    return ret + 1


def is_pair(pair_of_cards: tuple):
    if len(pair_of_cards) != 2:
        return False

    return get_card_value(pair_of_cards[0]) == get_card_value(pair_of_cards[1])


def get_card_value(card:str):
    """
    Get value of a card.
    NOTE: Ace card returns 11.
    """
    if card[0] in PICTURECARD:
        return 10
    if card[0] == "A":
        return 11
    if len(card) == 3:
        return 10
    return int(card[0])


def get_recommended_action(my_cards: tuple, dealer_upcard: str, surrender_allowed=False, split_allowed=False,
                           double_allowed=False, das=False) -> int:
    """
    Get recommended action for a player.

    Returns:
    int:
    -1 = nothing to recommend
    0 = stand
    1 = hit
    2 = double
    4 = split the pair
    8 = surrender
    """
    print("my_cards=%s dealer_upcard=%s" % (my_cards, dealer_upcard))

    my_sum = 0
    ace_count = 0
    for card in my_cards:
        card: str

        # if card is jack king queen (10)
        if card[0] in PICTURECARD:
            my_sum += 10

        elif card[0] == "A":
            ace_count += 1

        else:
            if len(card) == 2:
                # card is 2~9
                my_sum += int(card[0])
            else:
                # card is 10
                my_sum += 10
    # SOFT means player has ace. HARD means player has no ace.
    is_soft = ace_count > 0
    my_sum = get_sum_of_cards(my_sum, ace_count)

    if my_sum >= 21:
        return -1

    print("my_sum=%d ace_count=%d" % (my_sum, ace_count))

    dealer_index = get_card_value(dealer_upcard) - 2

    if surrender_allowed and (14 <= my_sum <= 16):
        if LATESURRENDER[16 - my_sum][dealer_index] == "8":
            return 8

    if split_allowed and is_pair(my_cards):
        card_value = get_card_value(my_cards[0])
        element = PAIRSPLITTING[11 - card_value][dealer_index]

        if element == "6" and das:
            return 4

        if element == "4":
            return 4

    if is_soft:
        element = SOFTTOTALS[20 - my_sum][dealer_index]
    else:
        if my_sum <= 7:
            return 1
        if my_sum >= 17:
            return 0
        tmp = HARDTOTALS[17 - my_sum]
        element = tmp[dealer_index]

    if element == "2":
        if double_allowed:
            return 2
        else:
            return 1

    if element == "3":
        if double_allowed:
            return 2
        else:
            return 0

    return int(element)

'''
model = YOLOCluster()

source = 0
if DEBUG_VIDEO:
    source = os.path.join(yoluster_dir, "yolo", "train_workspace", "extern_test_videos", "blackjack.mp4")

cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
count = 0

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model(frame)
        result = results[0]

        mparr = result.mparr
        print(mparr)
        if len(mparr) > 1:
            print("called")
            dealer_upcard = mparr[0][0]
            if len(mparr[1]) > 1:
                print("called")
                print("Recommended action:", get_recommended_action(mparr[1], dealer_upcard))

        annotated_frame = model.plotc(result)

        cv2.imshow("YOLOv8 Inference", annotated_frame)

        pkey = cv2.waitKey(1) & 0xFF

        if pkey == ord("q"):
            break

        if pkey == ord("p"):
            cv2.waitKey()

        if DEBUG_VIDEO:
            count += 3
            cap.set(cv2.CAP_PROP_POS_FRAMES, count)

    else:
        break

cap.release()
cv2.destroyAllWindows()
'''