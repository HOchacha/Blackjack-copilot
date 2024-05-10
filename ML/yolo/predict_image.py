from ultralytics import YOLO
import constants

print("import--")

IMAGE=r"videoframe_55508.png"
model=YOLO(constants.MODEL)
results = model(IMAGE)
result =results[0]
result.show()
