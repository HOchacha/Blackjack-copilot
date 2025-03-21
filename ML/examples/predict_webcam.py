import os
import sys

af = os.path.abspath(__file__)
di = os.path.dirname(af)
did = os.path.dirname(di)
sys.path.append(did)

# based on https://docs.ultralytics.com/modes/predict/#streaming-source-for-loop

from yoluster import YOLOCluster
import cv2

# Load the YOLOv8 model
model = YOLOCluster()

# Open the webcam
cap = cv2.VideoCapture(0)
count = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = model.plotc(results[0])

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        pkey = cv2.waitKey(1) & 0xFF

        # Break the loop if 'q' is pressed
        if pkey == ord("q"):
            break

        # pause if 'p' is pressed
        if pkey == ord("p"):
            cv2.waitKey()
        
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
