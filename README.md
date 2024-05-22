# Blackjack-copilot
Blackjack Copilot recommends a next action(e.g. Hit, stand) which has the highest winning rate.

It gets the card game board image as input, and gives an optimal action as output.

## ML training result
Object detection is based on YOLOv8s model.

It is trained for 13 epochs(10 epochs are pretrained by PD-Mera. [#](https://github.com/PD-Mera/Playing-Cards-Detection?tab=readme-ov-file#experiment-results)).

Training dataset has 10100 images.

![image](https://github.com/HOchacha/Blackjack-copilot/blob/f3d8c944122eb5a75284d99ed601b3c836a28c5c/ML/yolo/train_workspace/runs/detect/train/val_batch2_pred.jpg?raw=true)
![image](https://github.com/HOchacha/Blackjack-copilot/assets/70701690/d9f14337-2f5b-4b25-a261-2c169afcac5e)
![image](https://github.com/HOchacha/Blackjack-copilot/assets/70701690/ec2fb334-465c-49d4-a070-d4d8ad9e22e5)

## Reference
- [roboflow Playing Cards Dataset](https://universe.roboflow.com/augmented-startups/playing-cards-ow27d)
- [GitHub source of pretrained model](https://github.com/PD-Mera/Playing-Cards-Detection)
- [download link of pretrained model](https://drive.google.com/file/d/1AqZnW6dI6flFZvGxAn6A9apDNSviXZ5f/view?usp=share_link)
- [another GitHub repository which used the same pretrained model](https://github.com/noorkhokhar99/Playing-Cards-Detection-with-YoloV8)
- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com)
- [Blackjack playing sample YouTube video](https://www.youtube.com/watch?v=fbb5nFIjMn0)

