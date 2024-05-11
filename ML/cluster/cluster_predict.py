import sys
import os
import copy

imported = __name__ != "__main__"
if imported: sys.path.append(os.path.dirname(__file__))

import matcher
import math2

if imported: sys.path.remove(os.path.dirname(__file__))

def predict(yresult):
    xyxycpu = yresult.boxes.xyxy.cpu()
    height = yresult.orig_shape[0]
    width = yresult.orig_shape[1]
    cls = yresult.boxes.cls.cpu()

    n = len(xyxycpu)

    if n == 0:
        return []
    
    if n == 1:
        return [0]
    
    matcharr = matcher.match_cards(yresult)
    xyxycpu_clone = copy.deepcopy(xyxycpu)
    
    standard = min(width, height) * 0.16

    # -1 means it is not belong to any cluster yet
    ret = [-1] * n
    uniqueid = 0

    for i in range(n):

        # if i has no cluster, assign unique cluster
        if ret[i] == -1:
            ret[i] = uniqueid
            uniqueid += 1

        for j in range(i+1, n):
            distance = math2.get_distance2(xyxycpu[i], xyxycpu[j])

            threshold = standard
            # apply loose standard if two classes are equal
            if cls[i] == cls[j]:
                threshold = threshold * 1.4

            # if i and j should be in the same cluster
            if distance < threshold:
                if ret[j] != -1:
                    ret[i] = ret[j]
                else:
                    ret[j] = ret[i]
    
    return ret


