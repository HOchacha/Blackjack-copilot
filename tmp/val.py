import datetime
from ultralytics import YOLO
import time
import constants

# Load a model
model = YOLO(constants.MODEL)  # load a pretrained model

if __name__ == "__main__":
    STARTTIME=time.time()
    print("Start time: %s"%datetime.datetime.now())
    results = model.val()
    print("End time: %s"%datetime.datetime.now())
    print("%fs elapsed"%(time.time() - STARTTIME))
