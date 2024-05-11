import yolo_cluster_predict

path = __file__ + r"\..\yolo\images\1280x720.png"
result = yolo_cluster_predict.predict(path)
print(result)
result[0].show()
