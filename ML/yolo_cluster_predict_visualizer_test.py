import yolo_cluster_predict_visualizer
import yolo_cluster_predict
import util

imagesdir = __file__ + r"\..\yolo\images"
files = util.get_files(imagesdir)

for file in files:
    path = imagesdir + "\\" + file
    result = yolo_cluster_predict.predict(path)
    yolo_cluster_predict_visualizer.visualize(result)
