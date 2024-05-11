import yolo_predict

target = ["1280x720", "854x480", "640x360"]

for i in target:
    result = yolo_predict.predict(__file__ + r"\..\images\%s.png"%i)
    xyxyn_cpu = result.boxes.xyxyn.cpu()
    center_arr = yolo_predict.get_center2(xyxyn_cpu)
    print(center_arr)
    result.show()
