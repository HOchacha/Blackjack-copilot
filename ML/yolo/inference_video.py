import time
import cv2
import numpy as np
import torch
import pafy
import constants
from inference import get_model

print("import--")

def resize(frame):
    scale_percent = 40 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    frame_s = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return frame_s

def use_result(results, frame):
    print(results)
    
    cv2.imshow("Img", frame)
    return

model = get_model(model_id=constants.MODEL_ID)
print("get_model--")

def fromvideo():
    video = pafy.new(constants.URL)
    print("pafy--")
    best = video.getbestvideo(preftype="mp4")
    print("getbestvideo--")
    cap = cv2.VideoCapture(best.url)
    print("fromvideo--")
    return cap

cap = cv2.VideoCapture(constants.VIDEO)
#cap=fromvideo()
#cap = cv2.VideoCapture(0)

cudaa=torch.cuda.is_available()
print("cudaa=%s"%cudaa)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if cudaa:
        results = model.infer(frame, device=0)
    else:
        results = model.infer(frame)

    # # For Apple Silicon
    # if torch.backends.mps.is_available() :
    #     results = model(frame, device="mps") # Use MPS
    # else :
    #     results = model(frame)
    use_result(results, frame)

    key = cv2.waitKey(1)
    if key > 0:
        break

cap.release()
cv2.destroyAllWindows()
