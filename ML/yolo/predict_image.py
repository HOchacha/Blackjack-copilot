from ultralytics import YOLO
import constants

print("import--")

model=YOLO(constants.MODEL)
results = model(constants.IMAGE)
result =results[0]
result.show()
