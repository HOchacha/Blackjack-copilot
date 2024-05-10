from ultralytics import YOLO
import constants
print("import--")

im=r"test\videoframe_191642.png"

model=YOLO(constants.BEST)
results = model.predict(im, imgsz=(720,1280))
results[0].show()
