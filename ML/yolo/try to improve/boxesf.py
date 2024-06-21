from ultralytics.engine.results import Boxes
import torch

import plainf

def get_box_xyxy(box:Boxes) -> list:
    """
    Get box's xyxy. It is guaranteed that all coordinates are non-negative.
    """
    ret = [*map(float, box.xyxy[0])]
    
    assert len(ret) == 4
    for i in ret:
        assert i >= 0

    return ret

def get_cls(box:Boxes)->int:
    return int(box.cls)

def get_conf(box:Boxes)->float:
    """
    Return confidence (0, 1].
    """
    return float(box.conf)

def extract_best(parent:list, boxes:Boxes) -> Boxes:
    """
    Returns the best boxes.
    """

    # Use two pointer algorithm to do work on the boxes in the same set
    begin = 0
    end = 0
    n = boxes.shape[0]

    if n < 2:
        return boxes
    
    assert len(parent) == n

    for i in range(n):
        parent[i] = plainf.find(parent, i)

    boxarr = [*boxes]
    for i in range(n):
        boxarr[i].parent = parent[i]


    boxarr.sort(key = lambda x:x.conf, reverse = True)

    # The sort is in-place (i.e. the list itself is modified) and stable (i.e. the order of two equal elements is maintained).
    boxarr.sort(key = lambda x:x.parent)

    while begin < n and end < n:

        while end + 1 < n and boxarr[end].parent == boxarr[end + 1].parent:
            end += 1
        
        # begin=Beginning inclusive index of the same set
        # end=Ending inclusive index of the same set

        # d[cls] = (x,y)
        # x is a number of boxes for that class
        # y is sum of confidence.
        d = dict()

        for i in range(begin, end+1):
            box = boxarr[i]
            box:Boxes
            cls = get_cls(box)
            conf = get_conf(box)

            if cls not in d:
                d[cls] = [0,0]
            
            d[cls][0] += 1
            d[cls][1] += conf
        
        bestcls = plainf.argmax_dict(d)
        for i in range(begin, end+1):
            box = boxarr[i]
            cls = get_cls(box)
            if bestcls == cls:
                box.best = True

                # Only one box can be the best box,
                # which has the greatest confidence.
                # (It was previously sorted by confidence by stable)
                bestcls = -1

            else:
                box.best = False

        assert bestcls == -1

        begin = end + 1
        end = end + 1
    
    ret = None

    for box in boxarr:
        
        if ret == None and box.best:
            ret = box

            del box.best
            del box.parent

            continue

        if box.best:
            del box.best
            del box.parent

            ret = cat_boxes(ret, box)
    
    return ret

def cat_boxes(boxes0:Boxes, boxes1:Boxes) -> Boxes:
    """
    Concatenates two boxes into one boxes and returns it.
    Two boxes must have the same orig_shape.
    """
    assert boxes0.orig_shape == boxes1.orig_shape
    return Boxes(torch.cat((boxes0.data, boxes1.data)), boxes0.orig_shape)

def rotate_boxes(boxes:Boxes) -> Boxes:
    """
    Rotate boxes of result.
    """
    orig_shape = boxes.orig_shape
    data = boxes.data
    data = rotate_data(data, orig_shape)
    orig_shape = orig_shape[::-1]
    ret = Boxes(data, orig_shape)
    return ret

def union_boxes(boxes:Boxes, standard:float=0.45) -> list:
    """
    Union boxes whose IOU is greater than standard.
    Returns parent array.

    Union operation can be performed only once per each box.
    That means, chained boxes should not be considered as one object,
    like the situation (  [ )  { ]  < }  >
    """

    if standard >= 1:
        return

    n = boxes.shape[0]
    parent = [*range(n)]

    unioned = [False] * n

    xyxys = tuple(map(get_box_xyxy, boxes))

    for i in range(n-1):
        for j in range(i+1, n):

            a = xyxys[i]
            b = xyxys[j]

            iou = plainf.get_xyxy_IOU(a,b)
            if iou > standard:

                # if i has not been unioned, it can be unioned with another box.
                if not unioned[i]:

                    plainf.union(parent, i, j)
                    unioned[i] = True
    
    for i in range(n):
        parent[i] = plainf.find(parent, i)

    return parent

def rotate_data(data:torch.Tensor, orig_shape:tuple) -> torch.Tensor:
    """
    Rotate data 90 clockwise.
    """

    ret = data.clone()
    shape = ret.shape
    nrow = shape[0]

    for i in range(nrow):
        row = ret[i]
        rotate_row(row, orig_shape)

    return ret

def rotate_row(row:torch.Tensor, orig_shape:tuple) -> None:
    """
    Rotate a tensor row.
    """

    for j in {0,2}:
        x = float(row[j])
        y = float(row[j+1])
        rret = plainf.rotate(x, y, orig_shape)
        row[j] = rret[0]
        row[j+1] = rret[1]
    
    a,b,c,d = float(row[2]), float(row[1]), float(row[0]), float(row[3])
    row[0] = a
    row[1] = b
    row[2] = c
    row[3] = d

    assert a <= c
    assert b <= d