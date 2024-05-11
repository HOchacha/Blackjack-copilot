def get_distance3(width, height):
    return (width**2 + height**2)**0.5

def get_distance(x1, y1, x2, y2):
    return get_distance3(x1 - x2, y1 - y2)

def get_distance2(xy1, xy2):
    return get_distance(xy1[0], xy1[1], xy2[0], xy2[1])

def get_mid(pt1, pt2):
    x1 = pt1[0]
    y1 = pt1[1]
    x2 = pt2[0]
    y2 = pt2[1]
    return ((x1+x2)/2, (y1+y2)/2)

def get_center(xyxy):
    x1 = xyxy[0]
    y1 = xyxy[1]
    x2= xyxy[2]
    y2 =xyxy[3]
    return ((x1+x2)/2, (y1+y2)/2)

def get_center_arr(xyxyarr):
    return [*map(get_center, xyxyarr)]

def get_distance4(xyxy1, xyxy2):
    return get_distance2(get_center(xyxy1), get_center(xyxy2))
