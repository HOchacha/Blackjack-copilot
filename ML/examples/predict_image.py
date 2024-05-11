import os

di = os.path.dirname(os.path.abspath(__file__))
IMAGE = os.path.join(di, "..", "yolo", "train_workspace", "extern_test_images", "1280x720.png")
if not os.path.isfile(IMAGE):
    raise FileNotFoundError(IMAGE)

import sys
sys.path.append(os.path.dirname(di))

import yoluster
import cv2

model = yoluster.get_best_yolo_model()
results = yoluster.yolo_predict_with_image_path(model, IMAGE)
result = yoluster.yoluster_predict(results[0])
print(result)
im = yoluster.plot(results[0])
cv2.imshow("result", im)
cv2.waitKey()
cv2.destroyAllWindows()
