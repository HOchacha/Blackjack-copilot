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
