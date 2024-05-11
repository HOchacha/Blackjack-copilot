import matcher
import math2

def predict(yresult):
    xyxycpu = yresult.boxes.xyxy.cpu()
    height = yresult.orig_shape[0]
    width = yresult.orig_shape[1]

    n = len(xyxycpu)

    if n == 0:
        return []
    
    if n == 1:
        return [0]
    
    matchedxyarr = matcher.get_matched_xy_arr(yresult)
    
    standard = min(width, height) * 0.2

    # -1 means it is not belong to any cluster yet
    ret = [-1] * n
    uniqueid = 0

    for i in range(n):

        # if i has no cluster, assign unique cluster
        if ret[i] == -1:
            ret[i] = uniqueid
            uniqueid += 1

        for j in range(i+1, n):
            distance = math2.get_distance2(matchedxyarr[i], matchedxyarr[j])

            # if i and j should be in the same cluster
            if distance < standard:
                if ret[j] != -1:
                    ret[i] = ret[j]
                else:
                    ret[j] = ret[i]
    
    return ret


