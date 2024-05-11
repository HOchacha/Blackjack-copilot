from ultralytics import YOLO
import cv2

def get_model(model_path:str) -> YOLO:
    return YOLO(model_path)

def predict_by_image_path(model:YOLO, image_path:str):
    im = cv2.imread(image_path)
    return predict_matlike(model, im)

def predict_matlike(model:YOLO, matlike):
    height = matlike.shape[0]
    width = matlike.shape[1]
    imgsz = (int(height), int(width))
        
    # NOTE: "imgsz" parameter is NECESSARY to predict for image size other than 640x640.
    results = model.predict(matlike, imgsz=imgsz) 

    return results
