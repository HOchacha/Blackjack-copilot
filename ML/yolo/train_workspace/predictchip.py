import yolopredict
import os

thisdirname = os.path.dirname(__file__)

model_path = os.path.join(thisdirname, "runs", "detect", "chips", "weights", "best.pt")
video_path = os.path.join(thisdirname, "extern_test_videos", "blackjack.mp4")

yolopredict.YOLOPredict.predict_video(model_path, video_path)
