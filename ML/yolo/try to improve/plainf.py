def find(parent:list, x:int) -> int:
    """
    Find root node index of the tree.
    """
    if parent[x] == x:
        return x
    
    ret = find(parent, parent[x])
    parent[x] = ret
    return ret

def union(parent:list, x:int, y:int) -> int:
    """
    Union two nodes.
    """
    xp = find(parent, x)
    yp = find(parent, y)
    if yp < xp:
        parent[xp] = yp
    else:
        parent[yp] = xp
    
    return xp

def get_intersect_length(a:float,b:float,c:float,d:float)->float:
    if a > b:
        b, a = a, b
    
    if c > d:
        d, c = c, d
    
    if a > c:
        a, b, c, d = c, d, a, b
    
    if a != 0:
        a, b, c, d = 0, b-a, c-a, d-a

    # a == 0

    if c == 0:
        ret = min(b, d)

    elif b <= c:
        ret = 0
    
    elif d <= b:
        ret = d - c
    
    else:
        ret = max(0, b - c)
    
    assert ret >= 0

    return ret

def get_intersection_area(xyxy0:list, xyxy1:list) -> float:
    assert len(xyxy0)==4
    assert len(xyxy1)==4
    xl = get_intersect_length(xyxy0[0], xyxy0[2], xyxy1[0], xyxy1[2])
    yl = get_intersect_length(xyxy0[1], xyxy0[3], xyxy1[1], xyxy1[3])
    return xl * yl

def get_area(xyxy:list)->float:
    w = xyxy[2] - xyxy[0]
    h = xyxy[3] - xyxy[1]
    assert w > 0
    assert h > 0
    ret = w * h
    assert ret > 0
    return ret

def get_xyxy_IOU(xyxy0:list, xyxy1:list) -> float:
    """
    Returns real number that belongs to interval [0, 1].
    """

    ia = get_intersection_area(xyxy0, xyxy1)
    ua = get_area(xyxy0) + get_area(xyxy1) - ia

    assert ua > 0

    ret =  ia / ua

    if ret < 0 or ret > 1:
        raise AssertionError("ret=%f"%ret)

    return ret

def argmax_dict(d:dict) -> int:
    """
    Return class which has the greatest number of boxes.
    If there are multiple, higher sum of confidences is chosen.
    """
    first = True
    ret = -1

    for key in d:
        if first:
            first = False
            ret = key
            continue

        # number of box compare
        if d[key][0] > d[ret][0]:
            ret = key

        elif d[key][0] == d[ret][0]:

            # sum of confidence compare, if number of box are equal
            if d[key][1] > d[ret][1]:
                ret = key
    
    assert ret != -1
    return ret

def rotate(x:float,y:float,orig_shape:tuple)->tuple:
    """
    Rotate 90 clockwise.
    """
    
    assert len(orig_shape) == 2
    h = orig_shape[0]
    return (h - y, x)

def rotate_shape(orig_shape:tuple) -> tuple:
    """
    Returns rotated shape.
    """
    return orig_shape[::-1]


def shift(start:list, shape:tuple) -> int:
    """
    Shift a start point to a next point.
    If a start point gets outside, return 1; otherwise, 0.
    """

    if start[1] + 1 < shape[1]:
        start[1] += 1
        return 0
    
    if start[0] + 1 < shape[0]:
        start[0] += 1
        start[1] = 0
        return 0
    
    start[0] += 1
    start[1] = 0
    return 1

def is_inside(point:tuple, shape:tuple) -> bool:
    """
    Is point inside shape?
    """
    return 0 <= point[0] < shape[0] and 0 <= point[1] < shape[1]