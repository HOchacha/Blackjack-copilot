from ultralytics import YOLO
import constants
import cv2
print("import--")

im=cv2.imread(constants.IMAGE)
height=im.shape[0]
width=im.shape[1]
ratio=640/max(height, width)
dsize=(int(width*ratio), int(height*ratio))
im2=cv2.resize(im, dsize)
cv2.imshow("winname", im2)
cv2.waitKey()
cv2.destroyAllWindows()

model=YOLO(constants.BEST)
results = model(im2)
result =results[0]
result.show()
