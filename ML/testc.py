BEST = r"C:\Blackjack-copilot\ML\yolo\train_workspace\runs\detect\train\weights\best.pt"

# https://docs.ultralytics.com/modes/predict/#streaming-source-for-loop
import cv2
from ultralytics import YOLO
import cluster

# Load the YOLOv8 model
model = YOLO(BEST)

# Open the video file
cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)


        # Visualize the results on the frame
        # annotated_frame = results[0].plot()
        annotated_frame = cluster.visualize_by_yresult(results[0])

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
