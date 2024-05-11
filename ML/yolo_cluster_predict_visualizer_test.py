import yolo_cluster_predict_visualizer
import yolo_cluster_predict

result = yolo_cluster_predict.predict(__file__ + r"\..\yolo\images\1280x720.png")
yolo_cluster_predict_visualizer.visualize(result)
