import os
import sys
import cv2

yoluster_dir = os.path.join(__file__, "..", "..", "..", "ML")
yoluster_dir = os.path.abspath(yoluster_dir)
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

    (5, 2): ("Hit", None, None, None),
    (5, 3): ("Hit", None, None, None),
    (5, 4): ("Hit", None, None, None),
    (5, 5): ("Hit", None, None, None),
    (5, 6): ("Hit", None, None, None),
    (5, 7): ("Hit", None, None, None),
    (5, 8): ("Hit", None, None, None),
    (5, 9): ("Hit", None, None, None),
    (5, 10): ("Hit", None, None, None),
    (5, 11): ("Hit", None, None, None),

    (6, 2): ("Hit", None, None, None),
    (6, 3): ("Hit", None, None, None),
    (6, 4): ("Hit", None, None, None),
    (6, 5): ("Hit", None, None, None),
    (6, 6): ("Hit", None, None, None),
    (6, 7): ("Hit", None, None, None),
    (6, 8): ("Hit", None, None, None),
    (6, 9): ("Hit", None, None, None),
    (6, 10): ("Hit", None, None, None),
    (6, 11): ("Hit", None, None, None),

    (7, 2): ("Hit", None, None, None),
    (7, 3): ("Hit", None, None, None),
    (7, 4): ("Hit", None, None, None),
    (7, 5): ("Hit", None, None, None),
    (7, 6): ("Hit", None, None, None),
    (7, 7): ("Hit", None, None, None),
    (7, 8): ("Hit", None, None, None),
    (7, 9): ("Hit", None, None, None),
    (7, 10): ("Hit", None, None, None),
    (7, 11): ("Hit", None, None, None),

    (8, 2): ("Hit", None, None, "Split"),
    (8, 3): ("Hit", None, None, "Split"),
    (8, 4): ("Hit", None, None, "Split"),
    (8, 5): ("Hit", None, None, "Split"),
    (8, 6): ("Hit", None, None, "Split"),
    (8, 7): ("Hit", None, None, None),
    (8, 8): ("Hit", None, None, None),
    (8, 9): ("Hit", None, None, None),
    (8, 10): ("Hit", None, None, None),
    (8, 11): ("Hit", None, None, None),

    (9, 2): ("Hit", None, None, None),
    (9, 3): ("Double", None, None, None),
    (9, 4): ("Double", None, None, None),
    (9, 5): ("Double", None, None, None),
    (9, 6): ("Double", None, None, None),
    (9, 7): ("Hit", None, None, None),
    (9, 8): ("Hit", None, None, None),
    (9, 9): ("Hit", None, None, None),
    (9, 10): ("Hit", None, None, None),
    (9, 11): ("Hit", None, None, None),

    (10, 2): ("Double", None, None, None),
    (10, 3): ("Double", None, None, None),
    (10, 4): ("Double", None, None, None),
    (10, 5): ("Double", None, None, None),
    (10, 6): ("Double", None, None, None),
    (10, 7): ("Double", None, None, None),
    (10, 8): ("Double", None, None, None),
    (10, 9): ("Double", None, None, None),
    (10, 10): ("Hit", None, None, None),
    (10, 11): ("Hit", None, None, None),

    (11, 2): ("Double", None, None, None),
    (11, 3): ("Double", None, None, None),
    (11, 4): ("Double", None, None, None),
    (11, 5): ("Double", None, None, None),
    (11, 6): ("Double", None, None, None),
    (11, 7): ("Double", None, None, None),
    (11, 8): ("Double", None, None, None),
    (11, 9): ("Double", None, None, None),
    (11, 10): ("Double", None, None, None),
    (11, 11): ("Double", None, None, None),

    (12, 2): ("Hit", None, None, None),
    (12, 3): ("Hit", None, None, None),
    (12, 4): ("Stand", None, None, None),
    (12, 5): ("Stand", None, None, None),
    (12, 6): ("Stand", None, None, None),
    (12, 7): ("Hit", None, None, None),
    (12, 8): ("Hit", None, None, None),
    (12, 9): ("Hit", None, None, None),
    (12, 10): ("Hit", None, None, None),
    (12, 11): ("Hit", None, None, None),
    (13, 2): ("Stand", None, None, None),
    (13, 3): ("Stand", None, None, None),
    (13, 4): ("Stand", None, None, None),
    (13, 5): ("Stand", None, None, None),
    (13, 6): ("Stand", None, None, None),
    (13, 7): ("Hit", None, None, None),
    (13, 8): ("Hit", None, None, None),
    (13, 9): ("Hit", None, None, None),
    (13, 10): ("Hit", None, None, None),
    (13, 11): ("Hit", None, None, None),

    (14, 2): ("Stand", None, None, None),
    (14, 3): ("Stand", None, None, None),
    (14, 4): ("Stand", None, None, None),
    (14, 5): ("Stand", None, None, None),
    (14, 6): ("Stand", None, None, None),
    (14, 7): ("Hit", None, None, None),
    (14, 8): ("Hit", None, None, None),
    (14, 9): ("Hit", None, None, None),
    (14, 10): ("Hit", None, None, None),
    (14, 11): ("Hit", None, None, None),

    (15, 2): ("Stand", None, None, None),
    (15, 3): ("Stand", None, None, None),
    (15, 4): ("Stand", None, None, None),
    (15, 5): ("Stand", None, None, None),
    (15, 6): ("Stand", None, None, None),
    (15, 7): ("Hit", None, None, None),
    (15, 8): ("Hit", None, None, None),
    (15, 9): ("Hit", None, None, None),
    (15, 10): ("Surrender", None, None, None),
    (15, 11): ("Hit", None, None, None),

    (16, 2): ("Stand", None, None, None),
    (16, 3): ("Stand", None, None, None),
    (16, 4): ("Stand", None, None, None),
    (16, 5): ("Stand", None, None, None),
    (16, 6): ("Stand", None, None, None),

    (16, 7): ("Hit", None, None, None),
    (16, 8): ("Hit", None, None, None),
    (16, 9): (None, None, "Surrender", None),
    (16, 10): (None, None, "Surrender", None),
    (16, 11): (None, None, "Surrender", None),

    (17, 2): ("Stand", None, None, None),
    (17, 3): ("Stand", None, None, None),
    (17, 4): ("Stand", None, None, None),
    (17, 5): ("Stand", None, None, None),
    (17, 6): ("Stand", None, None, None),
    (17, 7): ("Stand", None, None, None),
    (17, 8): ("Stand", None, None, None),
    (17, 9): ("Stand", None, None, None),
    (17, 10): ("Stand", None, None, None),
    (17, 11): ("Stand", None, None, None),

    (18, 2): ("Stand", None, None, None),
    (18, 3): ("Stand", None, None, None),
    (18, 4): ("Stand", None, None, None),
    (18, 5): ("Stand", None, None, None),
    (18, 6): ("Stand", None, None, None),
    (18, 7): ("Stand", None, None, None),
    (18, 8): ("Stand", None, None, None),
    (18, 9): ("Stand", None, None, None),
    (18, 10): ("Stand", None, None, None),
    (18, 11): ("Stand", None, None, None),

    (19, 2): ("Stand", None, None, None),
    (19, 3): ("Stand", None, None, None),
    (19, 4): ("Stand", None, None, None),
    (19, 5): ("Stand", None, None, None),
    (19, 6): ("Stand", None, None, None),
    (19, 7): ("Stand", None, None, None),
    (19, 8): ("Stand", None, None, None),
    (19, 9): ("Stand", None, None, None),
    (19, 10): ("Stand", None, None, None),
    (19, 11): ("Stand", None, None, None),

    (20, 2): ("Stand", None, None, None),
    (20, 3): ("Stand", None, None, None),
    (20, 4): ("Stand", None, None, None),
    (20, 5): ("Stand", None, None, None),
    (20, 6): ("Stand", None, None, None),
    (20, 7): ("Stand", None, None, None),
    (20, 8): ("Stand", None, None, None),
    (20, 9): ("Stand", None, None, None),
    (20, 10): ("Stand", None, None, None),
    (20, 11): ("Stand", None, None, None),
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
