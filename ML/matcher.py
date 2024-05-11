import math2

def match_cards(yresult):
    xyxycpu = yresult.boxes.xyxy.cpu()
    height = yresult.orig_shape[0]
    width = yresult.orig_shape[1]
    cls = yresult.boxes.cls.cpu()

    n = len(xyxycpu)

    if n == 0:
        return []
    
    if n == 1:
        return [-1]
     
    standard = min(width, height) * 0.23

    # -1 means it is not matched to any symbol yet
    ret = [-1] * n
    uniqueid = 0

    for i in range(n):

        # if already matched, continue
        if ret[i] != -1: continue

        for j in range(i+1, n):
            if cls[i] != cls[j]: continue

            distance = math2.get_distance2(xyxycpu[i], xyxycpu[j])

            # if i and j should be in the same cluster
            if distance < standard:
                ret[i] = uniqueid
                ret[j] = uniqueid
                uniqueid += 1
    
    return ret

def get_pairs(matcharr):
    n = len(matcharr)
    ret = []

    for i in range(n):
        cls = matcharr[i]
        if cls == -1: continue


        j = matcharr.index(cls, i+1)
        ret.append((i,j))
        matcharr[i] = -1
        matcharr[j] = -1

    return ret

def getpairarr(yresult):
    return get_pairs(match_cards(yresult))

def do_mid_on_arr(xyarr, pairs):
    for i, j in pairs:
        midxy = math2.get_mid(xyarr[i], xyarr[j])
        xyarr[i] = midxy
        xyarr[j] = midxy

def get_matched_xy_arr(yresult):
    pairs = getpairarr(yresult)
    xyxycpu = yresult.boxes.xyxy.cpu()
    centerarr = math2.get_center_arr(xyxycpu)
    do_mid_on_arr(centerarr, pairs)
    return centerarr
