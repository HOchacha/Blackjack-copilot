import os

from ultralytics import YOLO

DIRNAME = os.path.dirname(__file__)

def new_yolo(model=None) -> YOLO:
    if model == None:
        model = os.path.join(DIRNAME, "yolov8s_playing_cards.pt")
    
        if not os.path.isfile(model):
            model = os.path.join(DIRNAME, "yolov8_playing_card_detect", "yolov8s_playing_cards.pt")

            if not os.path.isfile(model):
                raise FileNotFoundError(model)

    yolo = YOLO(model)
    return yolo