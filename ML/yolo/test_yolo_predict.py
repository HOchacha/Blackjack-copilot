import yolo_predict

result = yolo_predict.predict(__file__ + r"\..\images\1280x720.png")
print(result)
result.show()
