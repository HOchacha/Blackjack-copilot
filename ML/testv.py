BEST = r"C:\Blackjack-copilot\ML\yolo\train_workspace\runs\detect\train\weights\best.pt"
URL='https://www.youtube.com/watch?v=fbb5nFIjMn0'

import cv2
import pafy
import yolo
import cluster

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
    model=yolo.get_model(BEST)
    cap = fromvideo(url)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = yolo.predict_matlike(model,frame)
        yresult = results[0]
        vis = cluster.visualize_by_yresult(yresult)
        print(cluster.get_final_result_by_yresult(yresult))
        cv2.imshow("result", vis)
        if cv2.waitKey(1) > 0:
            break

visualize_video(URL)
