import datetime
from ultralytics import YOLO
import time

MODEL=r"runs\detect\train9\weights\last.pt"

# Load a model
model = YOLO(MODEL)  # load a pretrained model (recommended for training)

if __name__ == "__main__":
    # Train the model
    STARTTIME=time.time()
    print("Start time: %s"%datetime.datetime.now())
    results = model.val()
    print("End time: %s"%datetime.datetime.now())
    print("%fs elapsed"%(time.time() - STARTTIME))
