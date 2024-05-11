import cv2
from ultralytics.utils.plotting import Annotator

def visualize(yolo_cluster_predict_result):
    yresult, cresult = yolo_cluster_predict_result
    im = yresult.orig_img

    annotator = Annotator(im)
    box_index = 0
    for box in yresult.boxes:
        b = box.xyxy[0]
        c = box.cls
        class_index = int(c)
        class_name = yresult.names[class_index]
        s_conf = "%.2f"%float(box.conf[0].cpu())
        cluster = cresult[box_index]
        label = "%s %s %s"%(class_name, s_conf, cluster)
        annotator.box_label(b, label)
        box_index += 1

    im = annotator.result()

    cv2.imshow("yolo_cluster_predict_visualize_result", im)
    cv2.waitKey()
    cv2.destroyAllWindows()
