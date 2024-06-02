import os
import datetime
from ultralytics import YOLO
import cv2
import torch

class YOLOPredict:

    @staticmethod
    def get_device() -> int:
        if torch.cuda.is_available():
            return torch.cuda.current_device()
        return -1

    @staticmethod
    def get_now() -> datetime.datetime:
        return datetime.datetime.now()

    @staticmethod
    def print_now() -> None:
        print(YOLOPredict.get_now())

    @staticmethod
    def predict_video(model_path:str, video_path:str) -> None:
        # Load the YOLOv8 model
        model = YOLO(model_path)

        # Open the video file
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
