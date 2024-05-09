import cv2
from ultralytics import YOLO
import numpy as np
import torch
import pafy
print("import--")

MODEL=r"runs\detect\train6\weights\last.pt" # Input your path of model file
URL='https://www.youtube.com/watch?v=fbb5nFIjMn0'

def resize(frame):
    scale_percent = 40 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    frame_s = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return frame_s

def use_result(results, frame) :
    if (results and results[0]) :
        print("YOLOv8 Result = ", results[0].__dict__)
#         # print("YOLOv8 Result.boxes = ", results[0].boxes)
#         # print("YOLOv8 Result.boxes.xyxy = ", results[0].boxes.xyxy)
        bboxes = np.array(results[0].boxes.xyxy.cpu(), dtype="int")
        print("YOLOv8 Result.boxes.xyxy.cpu() = ", bboxes)
        classes = np.array(results[0].boxes.cls.cpu(), dtype="int")
        print("YOLOv8 Result.boxes.cls.cpu() = ", classes)
        names = results[0].names
        # print("Class names = ", names)
        pred_box = zip(classes, bboxes)
        for cls, bbox in pred_box :
            (x, y, x2, y2) = bbox
#             # print("bounding box (",x,y,x2,y2,") has class ", cls)
            print("bounding box (",x,y,x2,y2,") has class ", cls, " which is ", names[cls])
            cv2.rectangle(frame, (x,y), (x2,y2), (0,0,255), 2)
#                 # Display class of bounding box
#                 # cv2.putText(frame, str(cls), (x, y-5), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
            cv2.putText(frame, str(names[cls]), (x, y-5), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
    
    cv2.imshow("Img", frame)

    return

model = YOLO(MODEL)
print("YOLO--")

def fromvideo():
    video = pafy.new(URL)
    print("pafy--")
    best = video.getbest(preftype="mp4")
    cap = cv2.VideoCapture(best.url)
    print("fromvideo--")
    return cap

#cap = cv2.VideoCapture("dogs.mp4")
cap=fromvideo()
#cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

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
