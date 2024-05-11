BEST = r"C:\Blackjack-copilot\ML\yolo\train_workspace\runs\detect\train\weights\best.pt"
URL='https://www.youtube.com/watch?v=fbb5nFIjMn0'

import cv2
import pafy
import yolo
import cluster
import time

def fromvideo(url):
    video = pafy.new(url)
    print("pafy--")
    streams = video.videostreams
    chosen = streams[0]
    for stream in streams:
        x, y = stream.dimensions
        if y == 720:
            chosen = stream
    return chosen.url

def visualize_video(url):
    model=yolo.get_model(BEST)
    streamurl = fromvideo(url)
    results = model.predict(streamurl, stream=True, imgsz=(1280,720))
    for yresult in results:
        vis = cluster.visualize_by_yresult(yresult)
        print(cluster.get_final_result_by_yresult(yresult))

visualize_video(URL)
