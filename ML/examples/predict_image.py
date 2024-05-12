import os

di = os.path.dirname(os.path.abspath(__file__))
IMAGE = os.path.join(di, "..", "yolo", "train_workspace", "extern_test_images", "1280x720_2.png")
if not os.path.isfile(IMAGE):
    raise FileNotFoundError(IMAGE)

import sys
sys.path.append(os.path.dirname(di))

from yoluster import YOLOCluster
import cv2

model = YOLOCluster()
results = model(IMAGE)
for result in results:
    print(result.mparr)
    im = model.plotc(result)
    cv2.imshow("plotc", im)
    cv2.waitKey()

cv2.destroyAllWindows()
