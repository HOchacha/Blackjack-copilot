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

def get_mid_xy_list(result):
    xyxy = result.boxes.xyxy

    # to remove GPU CUDA device information
    xyxy = xyxy.cpu()

    mid_xy = []
    for i in xyxy:
        x1=i[0]
        y1=i[1]
        x2=i[2]
        y2=i[3]
        mid_xy.append(((x1+x2)/2, (y1+y2)/2))
    
    return mid_xy
