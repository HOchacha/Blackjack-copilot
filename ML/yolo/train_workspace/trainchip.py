import os

di = os.path.dirname(__file__)
DATA=os.path.join(di, "datasets", "chips-counter-V4-color.v1i.yolov8-obb", "data.yaml")

PRETRAINED_MODEL = os.path.join(di, "runs", "detect", "chips", "weights", "best.pt")



# If file not found, raise error.

if not os.path.isfile(PRETRAINED_MODEL):
    raise FileNotFoundError(PRETRAINED_MODEL)

if not os.path.isfile(DATA):
    raise FileNotFoundError(DATA)


import yolotrain

if __name__ == "__main__":
    yolotrain.YOLOTrain.train(DATA, PRETRAINED_MODEL, "chips")
