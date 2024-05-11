from yolo import yolo_predict
from cluster import cluster_predict

def predict(image_path):
    result = yolo_predict.predict(image_path)
    xyxyn_cpu = result.boxes.xyxyn.cpu()
    mid_arr = yolo_predict.get_center2(xyxyn_cpu)
    cresult = cluster_predict.predict(mid_arr)
    return (result, cresult)
