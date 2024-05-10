from ultralytics import YOLO
from ultralytics.engine.results import Results

model=YOLO(r"ML\runs\detect\train16\weights\best.pt")

def predict(source) -> Results:
    results = model.predict(source, imgsz=(720,1280))
    return results[0]

def get_mid_xy_list(result:Results) -> list:
    xyxy = result.boxes.xyxy.cpu()
    mid_xy = []
    for i in xyxy:
        x1=i[0]
        y1=i[1]
        x2=i[2]
        y2=i[3]
        mid_xy.append(((x1+x2)/2, (y1+y2)/2))
    # output: [(x1,y1),(x2,y2),...,(xn,yn)]
    return mid_xy

def main():
    ret = predict(r"C:\Blackjack-copilot\ML\yolo\test\1280x720.png")
    print("----------------------------------")
    print(ret)
    ret.show()
    print(ret.boxes)

if __name__ == "__main__":
    main()
