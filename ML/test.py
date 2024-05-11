BEST = r"C:\Blackjack-copilot\ML\yolo\train_workspace\runs\detect\train\weights\best.pt"
TEST_IMAGE = r"C:\Blackjack-copilot\ML\yolo\train_workspace\extern_test_images\1280x720.png"

import yolo
import cluster
import cv2

model = yolo.get_model(BEST)
yresults = yolo.predict_by_image_path(model, TEST_IMAGE)
yresult = yresults[0]
cresult = cluster.predict(yresult)
presult = cluster.get_plain_result(yresult, cresult)
print(presult)
cluster.match_result(presult)
print(presult)
dealer_index = cluster.get_dealer_index(yresult, cresult)
print(dealer_index)

im = cluster.visualize(yresult, cresult)
cv2.imshow("result", im)
cv2.waitKey()
