import datetime
from ultralytics import YOLO
import torch
import time
import constants

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

# Load a model
model = YOLO(constants.MODEL)  # load a pretrained model (recommended for training)

if __name__ == "__main__":
    print_now()
    # Train the model
    STARTTIME=time.time()
    device=get_device()
    print("device=%s"%device)
    results = model.train(
        task="detect", mode="train", data=constants.DATA,
        device=device, patience=2, name=constants.TRAIN_NAME, exist_ok=True,
        verbose=True
        )
    elapsed=time.time() - STARTTIME
    print("%s elapsed"%seconds2str(elapsed))
    print_now()

