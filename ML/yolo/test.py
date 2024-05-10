from ultralytics import YOLO
import os
import class_filter
import time
import random

MODEL=r"runs\detect\train14\weights\best.pt" # Input your path of model file
IMAGES_DIR=r"datasets\Playing Cards.0\test\images"


images=class_filter.get_files(IMAGES_DIR)
random.shuffle(images)
ni=len(images)
model=YOLO(MODEL)

for i in range(0, ni, 5):
    pathes = [os.path.join(IMAGES_DIR, image) for image in images[i:i+5]]
    results = model(pathes)
    
    for result in results:
        result.show()

    time.sleep(1)
