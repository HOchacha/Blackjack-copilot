import yolo_cluster_predict_visualizer
import yolo_cluster_predict
import cv2
import pafy

def fromvideo(url):
    video = pafy.new(url)
    print("pafy--")
    streams = video.videostreams
    chosen = streams[0]
    for stream in streams:
        x, y = stream.dimensions
        if y == 720:
            chosen = stream
    cap = cv2.VideoCapture(chosen.url)
    print("fromvideo--")
    return cap

def visualize_video(url):
    cap = fromvideo(url)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = yolo_cluster_predict.predict2(frame)
        yolo_cluster_predict_visualizer.visualize(result, 1)

