import os
from ultralytics.utils.plotting import Annotator
import numpy as np
import sys
from ultralytics import YOLO
import cv2

di = os.path.dirname(os.path.abspath(__file__))
BEST = os.path.join(di, "yolo", "train_workspace", "runs", "detect", "train", "weights", "best.pt")

def get_files(dir:str) -> list:
    dirlist = os.listdir(dir)
    ret = [f for f in dirlist if os.path.isfile(os.path.join(dir, f))]
    return ret

def get_distance_wh(width:float, height:float) -> float:
    return (width**2 + height**2)**0.5

def get_distance_xyxy(x1:float, y1:float, x2:float, y2:float) -> float:
    return get_distance_wh(x1 - x2, y1 - y2)

def get_distance_pt1_pt2(pt1:tuple, pt2:tuple) -> float:
    return get_distance_xyxy(pt1[0], pt1[1], pt2[0], pt2[1])

def get_mid_point_pt1_pt2(pt1:tuple, pt2:tuple) -> tuple:
    x1 = pt1[0]
    y1 = pt1[1]
    x2 = pt2[0]
    y2 = pt2[1]
    return ((x1+x2)/2, (y1+y2)/2)

def get_center_xyxy(xyxy:list) -> tuple:
    if len(xyxy) != 4:
        raise Exception("len(xyxy) is not 4. xyxy=%s"%xyxy)
    x1 = xyxy[0]
    y1 = xyxy[1]
    x2= xyxy[2]
    y2 =xyxy[3]
    return ((x1+x2)/2, (y1+y2)/2)

def get_center_arr_xyxy_arr(xyxy_arr:list) -> list:
    return [*map(get_center_xyxy, xyxy_arr)]

def match_cards(yresult, k=0.32) -> list:
    xyxycpu = yresult.boxes.xyxy.cpu()
    height = yresult.orig_shape[0]
    width = yresult.orig_shape[1]
    cls = yresult.boxes.cls.cpu()

    n = len(xyxycpu)

    if n == 0:
        return []
    
    if n == 1:
        return [-1]
     
    standard = min(width, height) * k

    # -1 means it is not matched to any symbol yet
    ret = [-1] * n
    uniqueid = 0

    for i in range(n-1):

        # if already matched, continue
        if ret[i] != -1: continue

        for j in range(i+1, n):
            if cls[i] != cls[j]: continue
            if ret[j] != -1: continue

            distance = get_distance_pt1_pt2(xyxycpu[i], xyxycpu[j])

            # if i and j should be in the same cluster
            if distance < standard:
                ret[i] = uniqueid
                ret[j] = uniqueid
                uniqueid += 1
                break
    
    # e.g. [-1 -1 0 -1 0 1 1 -1 -1 2 -1 -1 3 2 -1 3]
    return ret

def get_pairs_from_matcharr(matcharr:list) -> list:
    n = len(matcharr)
    if n < 2:
        return []

    ret = []

    try:
        for i in range(n-1):
            cls = matcharr[i]
            if cls == -1: continue

            j = matcharr.index(cls, i+1)
            ret.append((i,j))
            matcharr[i] = -1
            matcharr[j] = -1

    except ValueError:
        errmsg = "i=%d n=%d cls=%d matcharr=%s\n"%(i, n, cls, matcharr)
        sys.stderr.write(errmsg)
        raise

    # e.g. [(0 3) (4 9) (5 6) (7 10)]
    return ret

def get_pairs_with_yresult(yresult) -> list:
    matcharr = match_cards(yresult)
    return get_pairs_from_matcharr(matcharr)

def do_pair_on_arr(xyarr:list, match_pairs:list) -> None:
    for i, j in match_pairs:
        midxy = get_mid_point_pt1_pt2(xyarr[i], xyarr[j])
        xyarr[i] = midxy
        xyarr[j] = midxy

def get_matched_xy_arr(yresult) -> list:
    pairs = get_pairs_with_yresult(yresult)
    xyxycpu = yresult.boxes.xyxy.cpu()
    centerarr = get_center_arr_xyxy_arr(xyxycpu)
    do_pair_on_arr(centerarr, pairs)
    return centerarr

def compress_coords(arr:list) -> None:
    unique_coords = sorted(set(arr)) # O(n log n)
    coord_dict = {coord: i for i, coord in enumerate(unique_coords)} # O(n)
    compressed_coords = [coord_dict[coord] for coord in arr]
    
    for i in range(len(arr)):
        arr[i] = compressed_coords[i]

def cluster_predict(yresult, k=0.22) -> list:
    xyxycpu = yresult.boxes.xyxy.cpu()
    height = yresult.orig_shape[0]
    width = yresult.orig_shape[1]

    n = len(xyxycpu)

    if n == 0:
        return []
    
    if n == 1:
        return [0]
    
    matchedxyarr = get_matched_xy_arr(yresult)
    
    standard = min(width, height) * k

    # -1 means it is not belong to any cluster yet
    ret = [-1] * n
    uniqueid = 0

    for i in range(n):

        # if i has no cluster, assign unique cluster
        if ret[i] == -1:
            ret[i] = uniqueid
            uniqueid += 1

        for j in range(i+1, n):
            distance = get_distance_pt1_pt2(matchedxyarr[i], matchedxyarr[j])

            # if i and j should be in the same cluster
            if distance < standard:
                if ret[j] != -1:
                    ret[i] = ret[j]
                else:
                    ret[j] = ret[i]
        
    compress_coords(ret)



    # This code snippet does not affect on final result, but it is added for stability and testability
    dealer_index = get_dealer_index(yresult, ret)
    if dealer_index > 0:
        for i in range(n):
            if ret[i] == 0:
                ret[i] = dealer_index
            elif ret[i] == dealer_index:
                ret[i] = 0



    #e.g. [0 0 0 1 0 1 1 2 2 0 1 2 3]
    return ret

def get_number_of_clusters(cresult:list) -> int:
    if not cresult: return 0
    return max(cresult) + 1

def get_size_of_clusters(cresult:list) -> list:
    n_cluster = max(cresult) + 1
    ret = [0] * n_cluster
    for i in range(len(cresult)):
        ret[cresult[i]] += 1
    return ret

def get_dealer_index(yresult, cresult:list) -> int:
    n = len(cresult)
    if n == 0:
        return -1
    if n == 1:
        return 0
    
    n_clusters = max(cresult) + 1
    size_of_clusters = get_size_of_clusters(cresult)

    avgy = [0] * n_clusters

    matchedxyarr = get_matched_xy_arr(yresult)

    for i in range(n):
        y = matchedxyarr[i][1]
        clus = cresult[i]
        avgy[clus] += y
    
    for i in range(n_clusters):
        avgy[i] /= size_of_clusters[i]

    return np.argmin(avgy)

def get_class_of_box(box) -> int:
    return int(box.cls)

def get_plain_result(yresult, cresult:list) -> list:
    boxes = yresult.boxes

    class_names = yresult.names
    nclusters = get_number_of_clusters(cresult)
    ret = [None] * nclusters
    for i in range(nclusters):
        ret[i] = tuple()

    n = len(cresult)

    for i in range(n):
        box = boxes[i]
        cls = get_class_of_box(box)
        nameofclass = class_names[cls]
        cluster_idx = cresult[i]
        ret[cluster_idx] += (nameofclass,)

    return ret

def get_count_dict(bagg:tuple) -> dict:
    ret = dict()
    for i in bagg:
        if i not in ret:
            ret[i] = 1
        else:
            ret[i] += 1
    
    #e.g. {"KH": 2, "9S": 1, "6D": 3}
    return ret

def match_result(plain_result) -> None:
    n = len(plain_result)

    for i in range(n):
        clust = plain_result[i]
        count_dict = get_count_dict(clust)
        newtuple = tuple()
        for key in count_dict:
            count = count_dict[key]
            card_count = (count + 1) // 2
            newtuple += (key,) * card_count
        plain_result[i] = newtuple

def visualize(yresult, cresult:list):
    im = yresult.orig_img

    annotator = Annotator(im)
    box_index = 0
    for box in yresult.boxes:
        b = box.xyxy[0]
        c = box.cls
        class_index = int(c)
        class_name = yresult.names[class_index]
        s_conf = "%.2f"%float(box.conf[0].cpu())
        cluster = cresult[box_index]
        label = "%s %s %s"%(class_name, s_conf, cluster)
        annotator.box_label(b, label)
        box_index += 1

    im = annotator.result()
    return im

def plot(yresult):
    cresult = cluster_predict(yresult)
    return visualize(yresult, cresult)

def yoluster_predict(yresult) -> list:
    cresult = cluster_predict(yresult)
    presult = get_plain_result(yresult, cresult)
    match_result(presult)
    dealer_index = get_dealer_index(yresult, cresult)

    if dealer_index > 0:
        # Make dealer be always at index 0
        presult[0], presult[dealer_index] = presult[dealer_index], presult[0]
        dealer_index = 0

    return presult

def get_model(model_path:str) -> YOLO:
    return YOLO(model_path)

def get_best_yolo_model() -> YOLO:
    return get_model(BEST)

def yolo_predict_with_image_path(model:YOLO, image_path:str):
    im = cv2.imread(image_path)
    return yolo_predict_m(model, im)

def yolo_predict_m(model:YOLO, matlike):
    height = matlike.shape[0]
    width = matlike.shape[1]
    imgsz = (int(height), int(width))
        
    # NOTE: "imgsz" parameter is NECESSARY to predict for image size other than 640x640.
    results = model.predict(matlike, imgsz=imgsz) 

    return results