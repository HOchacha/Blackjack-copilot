BEST = r"C:\Blackjack-copilot\ML\yolo\train_workspace\runs\detect\train\weights\best.pt"
VIDEO_PATH = r"C:\Blackjack-copilot\ML\yolo\train_workspace\extern_test_videos\blackjack.mp4"

import cv2
import yolo
import cluster

def visualize_video():
    model=yolo.get_model(BEST)
    results = model.predict(VIDEO_PATH, stream=True, imgsz=(720,1280))
    for yresult in results:
        vis = cluster.visualize_by_yresult(yresult)
        print(cluster.get_final_result_by_yresult(yresult))
        cv2.imshow("winname", vis)
        if cv2.waitKey(1) > 0: break

visualize_video()
