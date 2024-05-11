import os

di = os.path.dirname(__file__)
DATA=os.path.join(di, "datasets", "Playing Cards.v3-original_raw-images.yolov8", "data.yaml")
PRETRAINED_MODEL = os.path.join(di, "pretrained_models", "yolov8s_playing_cards.pt")
if not os.path.isfile(PRETRAINED_MODEL):
    raise FileNotFoundError(PRETRAINED_MODEL)
if not os.path.isfile(DATA):
    raise FileNotFoundError(DATA)

import datetime
from ultralytics import YOLO
import torch
import time

def seconds2str(seconds:float) -> str:
    hours = seconds/60/60
    if seconds < 60:
        return "%ds (%f hours)"%(seconds, hours)
    return "%dm %ds (%f hours)"%(seconds//60, seconds%60, hours)

def get_device():
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    return None

def print_now():
    print(datetime.datetime.now())

model = YOLO(PRETRAINED_MODEL)

if __name__ == "__main__":
    print_now()
    starttime=time.time()
    device=get_device()
    print("device=%s"%device)
    results = model.train(task="detect", mode="train", data=DATA, device=device, patience=2, verbose=True)
    elapsed=time.time() - starttime
    print("%s elapsed"%seconds2str(elapsed))
    print_now()
