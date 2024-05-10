import yolo_predict
import cluster_predict

yolopredret = yolo_predict.predict(r"C:\Blackjack-copilot\ML\test\1280x720.png")
print(yolopredret)
midxyarr = yolo_predict.get_mid_xy_list(yolopredret)
print(midxyarr)
cluspredret = cluster_predict.cluster(midxyarr)
print(cluspredret)
