import os
import sys
import cv2

yoluster_dir = "C:/Users/user/PycharmProjects/Blackjack_Copilot"
sys.path.append(yoluster_dir)

from yoluster import YOLOCluster

blackjack_copilot_strategy = {
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

    (16, 9): (None, None, "Surrender", None),
    (16, 10): (None, None, "Surrender", None),
    (16, 11): (None, None, "Surrender", None),

    (8, 2): (None, None, None, "Split"),
    (8, 3): (None, None, None, "Split"),
    (8, 4): (None, None, None, "Split"),
    (8, 5): (None, None, None, "Split"),
    (8, 6): (None, None, None, "Split"),
    (10, 10): (None, None, None, "Split"),
}

model = YOLOCluster()

cap = cv2.VideoCapture(0)
count = 0

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model(frame)

        annotated_frame = model.plotc(results[0])

        cv2.imshow("YOLOv8 Inference", annotated_frame)

        pkey = cv2.waitKey(1) & 0xFF

        if pkey == ord("q"):
            break

        if pkey == ord("p"):
            cv2.waitKey()

    else:
        break

cap.release()
cv2.destroyAllWindows()
