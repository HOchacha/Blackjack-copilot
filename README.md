# Blackjack-copilot
Blackjack Copilot recommends a next action(e.g. hit, stand) which has the highest winning rate, that is, optimal for a player.

# To run the code
first do 
`git clone https://github.com/HOchacha/Blackjack-copilot`
then install the dependency
`pip install ultralytics opency-python numpy torch PyQt5`
Run
`python3 ./app/util/main.app.py`

## Description
It gets the card game board image as input, and gives an optimal action as output. Getting a video as input is of course possible since a video is composed of a plenty of continous image frames.

## TODO
- [x] Train a YOLOv8 object detection model
- [x] Cluster cards for a dealer and players
- [x] Implement a lookup table of optimal actions
- [x] Recommend action in the case that player has more than two cards
- [x] Make Blackjack Copilot work for video, webcam or stream

## Machine Learning Training Result
Object detection is based on YOLOv8s model.

It is trained for 13 epochs(10 epochs are pretrained by [PD-Mera](https://github.com/PD-Mera/Playing-Cards-Detection?tab=readme-ov-file#experiment-results)).

The training dataset has 10100 original images (except augmented).

![image](/ML/yolo/train_workspace/runs/detect/train/val_batch2_pred.jpg)

![image](/app/util/Lookup_Algorithm_Output/testing.jpg)

![image](/app/util/Lookup_Algorithm_Output/testing2.jpg)

## Reference
- [roboflow Playing Cards Dataset](https://universe.roboflow.com/augmented-startups/playing-cards-ow27d)
- [GitHub repository as source of pretrained model](https://github.com/PD-Mera/Playing-Cards-Detection)
- [download link of pretrained model](https://drive.google.com/file/d/1AqZnW6dI6flFZvGxAn6A9apDNSviXZ5f/view?usp=share_link)
- [another GitHub repository which contains the same pretrained model](https://github.com/noorkhokhar99/Playing-Cards-Detection-with-YoloV8) (Jan 27, 2023)
- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com)
- [Blackjack playing video](https://www.youtube.com/watch?v=fbb5nFIjMn0) (not demo of this project)
- [opencv-python](https://pypi.org/project/opencv-python)
