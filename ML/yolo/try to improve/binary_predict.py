from ultralytics import YOLO
import os
import cv2

dirname = os.path.dirname(__file__)
model_path = os.path.join(dirname, "yolov8n_playing_cards_binary.pt")
image_path = os.path.join(dirname, "0.jpg")

model = YOLO(model_path)

frame = cv2.imread(image_path)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

results = model(frame)
result = results[0]
plot = result.plot()

while plot.shape[0] > 1000 or plot.shape[1] > 1900:
    plot = cv2.resize(plot, (plot.shape[1]//2, plot.shape[0]//2))

cv2.imshow("plot", plot)
cv2.waitKey()
