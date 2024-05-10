from ultralytics import YOLO
import os
import class_filter
import time
import random
import constants

IMAGES_DIR=constants.DATA+r"\..\test\images"

images=class_filter.get_files(IMAGES_DIR)
random.shuffle(images)
ni=len(images)
model=YOLO(constants.MODEL)

for i in range(0, ni, 5):
    pathes = [os.path.join(IMAGES_DIR, image) for image in images[i:i+5]]
    results = model(pathes)
    
    for result in results:
        result.show()

    time.sleep(1)
