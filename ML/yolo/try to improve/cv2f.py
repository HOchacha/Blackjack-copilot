import os
from typing import Callable, Union
import cv2
from cv2.typing import MatLike
import numpy as np
import skimage

import plainf

DIRNAME = os.path.dirname(__file__)
BLACK = 0
ESC = 40
GRAY = 180
WHITE = 255
DX=(0,0,1,-1)
DY=(1,-1,0,0)

def imshow(im:MatLike, delay:int=0, winname:str="imshow") -> int:
    """
    Show image. Return key waited.
    """
    flag = False

    while im.shape[1] < 800 and im.shape[0] < 400:
        im = cv2.resize(im, (im.shape[1]<<1, im.shape[0]<<1))
        flag = True
    
    if not flag:
        while im.shape[1] > 1600 or im.shape[0] > 800:
            im = cv2.resize(im, (im.shape[1]>>1, im.shape[0]>>1))

    cv2.imshow(winname, im)
    return cv2.waitKey(delay)

def rotate(orig_img:MatLike) -> MatLike:
    """
    Returns 90-clockwise-rotated image.
    """
    return cv2.rotate(orig_img, cv2.ROTATE_90_CLOCKWISE)


def thresh(im:MatLike, blurksize=5, maxpool=True, invert=False) -> np.ndarray:
    """
    Preprocess an predict cropped box image.
    Suit and rank part is denoted as 255 (white).
    The other, such as background, is denoted as 0 (black).
    """
    # Convert it to gray
    ret = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Reduce the noise to avoid false detection
    if blurksize > 0:
        ret = cv2.medianBlur(ret, 5)

    ret = cv2.threshold(ret, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    if maxpool:
        ret = skimage.measure.block_reduce(ret, (2,2), np.max)

    if invert:
        ret = cv2.bitwise_not(ret)

    return ret

def mov_start(mat:MatLike, start:list) -> int:
    plainf.shift(start, mat.shape)

    while (start[0] < mat.shape[0]) and (mat[start[0],start[1]] != BLACK):
        plainf.shift(start, mat.shape)
    
    if start[0] >= mat.shape[0]:
        return 1
    
    return 0

def do_start(mat:MatLike, start:list) -> tuple[int, MatLike]:
    """
    DFS.
    If a confined space is not found, return (0, mat).
    If a confined space is found, return (the number of visit, mat).
    """

    stack = []
    stack.append(start)
    nvisit = 0

    while stack:
        top = stack.pop()

        color = mat[top[0], top[1]]
        if color != BLACK:
            continue
        
        nvisit += 1
        mat[top[0], top[1]] = GRAY

        for i in range(4):
            nextpoint = (top[0]+DY[i], top[1]+DX[i])

            # if nextpoint is out of bound
            if not plainf.is_inside(nextpoint, mat.shape):
                # Escaping out of the box is possible!
                # So it is not confined.
                # Mark the path escapable.
                # Replace GRAY to ESC.
                mat = np.where(mat == GRAY, ESC, mat)
                return (0, mat)
            
            nextcolor = mat[nextpoint[0], nextpoint[1]]

            # if escape possible
            if nextcolor == ESC:
                mat = np.where(mat == GRAY, ESC, mat)
                return (0, mat)
        
            # if already visited
            if nextcolor == GRAY:
                continue

            # if it is wall
            if nextcolor == WHITE:
                continue

            assert nextcolor == BLACK
            stack.append(nextpoint)

    return (nvisit, mat)

def confine(mat:MatLike) -> tuple[int, MatLike]:
    """
    If there is any confined space, return its area and colored matrix; otherwise, return [0, colored_matrix].
    """
    start = [0,0]

    while start[0] < mat.shape[0]:
        nvisit, mat = do_start(mat, start)
        if nvisit > 0:
            return (nvisit, mat)
        mov_start(mat, start)
    
    return (0, mat)

def is_there_any_confined_space(mat:MatLike) -> bool:
    pre = thresh(mat)
    area, mat = confine(pre)
    return area > 0

def get_red_mask(hsv:MatLike):
    # Source: https://cvexplained.wordpress.com/2020/04/28/color-detection-hsv
    # lower boundary RED color range values; Hue (0 - 10)
    lower1 = np.array([0, 100, 20])
    upper1 = np.array([10, 255, 255])
    lower_mask = cv2.inRange(hsv, lower1, upper1)
    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([160,100,20])
    upper2 = np.array([179,255,255])
    upper_mask = cv2.inRange(hsv, lower2, upper2)
    full_mask = lower_mask + upper_mask
    return full_mask

def whitemask(src:MatLike) -> MatLike:
    srcc = src.copy()
    image_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    # lower boundary WHITE color range values; Value (127 - 255)
    lower1 = np.array([0, 0, 127])
    upper1 = np.array([255, 255, 255])
    
    lower_mask = cv2.inRange(image_hsv, lower1, upper1)
    
    mask_where = np.where(lower_mask)
    srcc[mask_where] = (255,255,255)
    return srcc

def do_cam(f:Callable[[MatLike], int], index:int=0, apiPreference:int=cv2.CAP_DSHOW) -> int:
    ret = 0
    cap = cv2.VideoCapture(index, apiPreference)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            fret = f(frame)
            if fret != 0:
                ret = fret
                break

        else:
            # Break the loop if the end of the video is reached
            ret = 2
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
    return ret

def do_cam2(mask_func:Callable[[MatLike], MatLike]):
    def f(frame):
        result = mask_func(frame)

        # Display the annotated frame
        wkey = imshow(result, 444)

        # Break the loop if 'q' is pressed
        if wkey == ord('q'):
            return 1
        
        # Pause the loop if 'p' is pressed
        if wkey == ord("p"):
            cv2.waitKey()
        
        return 0
    
    do_cam(f)

def imread(source:Union[str, MatLike]=None):
    if source == None:
        source = os.path.join(DIRNAME, "images", "0.jpg")
        
        if not os.path.isfile(source):
            raise FileNotFoundError(source)
    
    if isinstance(source, str):
        source = cv2.imread(source)
    
    return source

def do_mask(mask_func:Callable[[MatLike], MatLike]):
    im = imread()
    imshow(mask_func(im))
    do_cam2(mask_func)
