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
