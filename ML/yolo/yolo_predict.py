from ultralytics import YOLO
import cv2

model=YOLO(r"ML\yolo\runs\detect\train\weights\best.pt")

def predict(source):
    im = cv2.imread(source)
    height = im.shape[0]
    width = im.shape[1]
    imgsz = (int(height), int(width))
    
    # NOTE: "imgsz" parameter is NECESSARY to predict for image size other than 640x640.
    results = model.predict(im, imgsz=imgsz) 

    return results[0]

def get_center(xyxyn):
    x1 = xyxyn[0]
    y1 = xyxyn[1]
    x2= xyxyn[2]
    y2 =xyxyn[3]
    return ((x1+x2)/2, (y1+y2)/2)

def get_center2(xyxyn_cpu):
    return [*map(get_center, xyxyn_cpu)]
