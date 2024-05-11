from yolo import yolo_predict
from cluster import cluster_predict

def predict(image_path):
    result = yolo_predict.predict(image_path)
    cresult = cluster_predict.predict(result)
    return (result, cresult)
