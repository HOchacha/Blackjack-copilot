import ultraf
import yolof
import cv2

im = cv2.imread(__file__+"/../0.jpg")

model = yolof.new_yolo()
result = ultraf.predict_best(model, im)
result.show()
