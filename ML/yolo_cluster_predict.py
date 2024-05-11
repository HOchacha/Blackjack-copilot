from yolo import yolo_predict
from cluster import cluster_predict

def predict(image_path):
    result = yolo_predict.predict(image_path)
    xyxyn_cpu = result.boxes.xyxy.cpu()
    height = result.orig_shape[0]
    width = result.orig_shape[1]
    mid_arr = yolo_predict.get_center2(xyxyn_cpu)
    cresult = cluster_predict.predict(mid_arr, height, width)
    return (result, cresult)
