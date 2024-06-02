import ML.yolo.train_workspace.yolopredict as yolopredict
import os

thisdirname = os.path.dirname(__file__)
os.chdir(thisdirname)

model_path = "yolov8n.pt"
video_path = os.path.join(thisdirname, "extern_test_videos", "blackjack.mp4")

yolopredict.YOLOPredict.predict_video(model_path, video_path)
