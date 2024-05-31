import os

di = os.path.dirname(__file__)
os.chdir(di)

PRETRAINED_MODEL = os.path.join(di, "runs", "detect", "chips", "weights", "best.pt")



# If file not found, raise error.

if not os.path.isfile(PRETRAINED_MODEL):
    raise FileNotFoundError(PRETRAINED_MODEL)



import datetime
from ultralytics import YOLO
import cv2
import torch
import time

def seconds2str(seconds:float) -> str:
    hours = seconds/60/60
    if seconds < 60:
        return "%ds (%f hours)"%(seconds, hours)
    return "%dm %ds (%f hours)"%(seconds//60, seconds%60, hours)

def get_device():
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    return None

def print_now():
    print(datetime.datetime.now())

model = YOLO(PRETRAINED_MODEL)

if __name__ == "__main__":
    print_now()
    starttime=time.time()
    device=get_device()
    print("device=%s"%device)



    # Load the YOLOv8 model
    model = YOLO(PRETRAINED_MODEL)

    # Open the video file
    video_path = os.path.join(di, "extern_test_videos", "blackjack.mp4")
    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()



    elapsed=time.time() - starttime
    print("%s elapsed"%seconds2str(elapsed))
    print_now()
