from inference import get_model
import constants
import cv2

def resize(frame):
    scale_percent = 40 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    frame_s = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return frame_s

model = get_model(model_id=constants.MODEL_ID)
results = model.infer(constants.IMAGE)
im = cv2.imread(constants.IMAGE)
result = results[0]
predictions=result.predictions
for pred in predictions:
    pt1=(int(pred.x), int(pred.y))
    pt2=(int(pred.x+pred.width), int(pred.y+pred.height))
    cv2.rectangle(im, pt1, pt2, (0,0,255), 2)
    cv2.putText(im, pred.class_name, pt1, cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

print(results)

im = resize(im)
cv2.imshow("result", im)
cv2.waitKey()
