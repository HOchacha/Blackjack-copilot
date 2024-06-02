import datetime
from ultralytics import YOLO
import torch
import time
import os

class YOLOTrain:

    @staticmethod
    def seconds2str(seconds:float) -> str:
        hours = seconds/60/60
        if seconds < 60:
            return "%ds (%f hours)"%(seconds, hours)
        return "%dm %ds (%f hours)"%(seconds//60, seconds%60, hours)

    @staticmethod
    def get_device() -> int:
        if torch.cuda.is_available():
            return torch.cuda.current_device()
        return -1

    @staticmethod
    def print_now() -> None:
        print(datetime.datetime.now())

    @staticmethod
    def train(dataset_path:str, pretrained_model_path:str, name:str):
        di = os.path.dirname(__file__)
        os.chdir(di)

        YOLOTrain.print_now()
        starttime=time.time()
        device=YOLOTrain.get_device()
        if device != -1:
            print("device=%s"%device)


        model = YOLO(pretrained_model_path)

        if device != -1:
            model.train(task="detect", mode="train", data=dataset_path, device=device, patience=2, verbose=True,
                                name=name, exist_ok=True)
        else:
            model.train(task="detect", mode="train", data=dataset_path, patience=2, verbose=True,
                                name=name, exist_ok=True)
            
        elapsed = time.time() - starttime
        print("%s elapsed"%YOLOTrain.seconds2str(elapsed))
        YOLOTrain.print_now()
