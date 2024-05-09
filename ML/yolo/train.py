from ultralytics import YOLO
import time

MODEL=r"runs\detect\train9\weights\last.pt"
DATA=r"C:\Blackjack-copilot\ML\yolo\datasets\Playing Cards.0\data.yaml"

# Load a model
model = YOLO(MODEL)  # load a pretrained model (recommended for training)

if __name__ == "__main__":
    # Train the model
    STARTTIME=time.time()
    results = model.train(data=DATA, epochs=2, device=0)
    print("%fs elapsed"%(time.time() - STARTTIME))
