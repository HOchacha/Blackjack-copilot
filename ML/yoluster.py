import os
from pathlib import Path
from typing import Union
import torch
import ultralytics
import ultralytics.engine
import ultralytics.engine.results
from ultralytics.utils.plotting import Annotator
import numpy as np
import sys
from ultralytics import YOLO
import cv2

class YOLOCluster(YOLO):
    
    def __init__(
        self,
        model: Union[str, Path] = None,
        task: str = None,
        verbose: bool = False,
    ) -> None:
        if model == None:
            model = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "yolo", "train_workspace", "runs", "detect", "train", "weights", "best.pt")
        super().__init__(model, task, verbose)

    C_RATIO = 0.22
    M_RATIO = 0.35

    def predict(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        predictor=None,
        **kwargs,
    ) -> list:
        ret = None

        if isinstance(source, str):
            isfile = os.path.isfile(source)
            if isfile:
                im = cv2.imread(source)
                h = im.shape[0]
                w = im.shape[1]
                imgsz = (int(h), int(w))
                ret = super().predict(source, stream, predictor, imgsz=imgsz, **kwargs)
        
        elif isinstance(source, np.ndarray):
            h = source.shape[0]
            w = source.shape[1]
            imgsz = (int(h), int(w))
            ret = super().predict(source, stream, predictor, imgsz=imgsz, **kwargs)

        if ret == None:
            ret = super().predict(source, stream, predictor, **kwargs)
        
        for i in ret:
            YOLOCluster._predictc(i)
        
        return ret
    
    @staticmethod
    def _get_pairs(matcharr:list) -> list:
        "[-1 -1 0 -1 0 1 2 1 2 -1] -> [(2,4) (5,7) (6,8)]"
        matcharr = matcharr[:]
        n = len(matcharr)

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

        return ret

    @staticmethod
    def _match_cards(yresult) -> None:
        xyxycpu = yresult.boxes.xyxy.cpu()
        height = yresult.orig_shape[0]
        width = yresult.orig_shape[1]
        cls = yresult.boxes.cls.cpu()

        n = len(xyxycpu)

        if n == 0:
            ret = []
        
        elif n == 1:
            ret = [-1]
        
        else:
            standard = min(width, height) * YOLOCluster.M_RATIO

            # -1 means it is not matched to any symbol yet
            ret = [-1] * n
            uniqueid = 0
            
            for i in range(n-1):
                # if already matched, continue
                if ret[i] != -1: continue

                for j in range(i+1, n):
                    if cls[i] != cls[j]: continue
                    if ret[j] != -1: continue

                    distance = YOLOCluster._get_distance_pt1_pt2(xyxycpu[i], xyxycpu[j])

                    # if i and j should be in the same cluster
                    if distance < standard:
                        ret[i] = uniqueid
                        ret[j] = uniqueid
                        uniqueid += 1
                        break
        
        # ret e.g. [-1 -1 0 -1 0 1 1 -1 -1 2 -1 -1 3 2 -1 3]
        yresult.matcharr = ret
    
    @staticmethod
    def _get_pairs_y(yresult) -> None:
        YOLOCluster._match_cards(yresult)
        pairs = YOLOCluster._get_pairs(yresult.matcharr)
        yresult.pairs = pairs

    @staticmethod
    def _get_center(xyxy:list) -> tuple:
        if len(xyxy) != 4:
            raise Exception("len(xyxy) is not 4. xyxy=%s"%xyxy)
        x1 = xyxy[0]
        y1 = xyxy[1]
        x2 = xyxy[2]
        y2 = xyxy[3]
        return ((x1+x2)/2, (y1+y2)/2)

    @staticmethod
    def _get_center_arr(xyxy_arr:list) -> list:
        return [*map(YOLOCluster._get_center, xyxy_arr)]
    
    @staticmethod
    def _get_mid_point_pt1_pt2(pt1:tuple, pt2:tuple) -> tuple:
        assert len(pt1) == 2
        assert len(pt2) == 2
        x1 = pt1[0]
        y1 = pt1[1]
        x2 = pt2[0]
        y2 = pt2[1]
        return ((x1+x2)/2, (y1+y2)/2)
    
    @staticmethod
    def _do_pair_on_arr(xyarr:list, match_pairs:list) -> None:
        for i, j in match_pairs:
            midxy = YOLOCluster._get_mid_point_pt1_pt2(xyarr[i], xyarr[j])
            xyarr[i] = midxy
            xyarr[j] = midxy

    @staticmethod
    def _get_mxyarr(yresult) -> None:
        YOLOCluster._get_pairs_y(yresult)
        xyxycpu = yresult.boxes.xyxy.cpu()
        centerarr = YOLOCluster._get_center_arr(xyxycpu)
        YOLOCluster._do_pair_on_arr(centerarr, yresult.pairs)
        yresult.mxyarr = centerarr
    
    @staticmethod
    def _get_distance_wh(width:float, height:float) -> float:
        return (width**2 + height**2)**0.5

    @staticmethod
    def _get_distance_xyxy(x1:float, y1:float, x2:float, y2:float) -> float:
        return YOLOCluster._get_distance_wh(x1 - x2, y1 - y2)

    @staticmethod
    def _get_distance_pt1_pt2(pt1:tuple, pt2:tuple) -> float:
        return YOLOCluster._get_distance_xyxy(pt1[0], pt1[1], pt2[0], pt2[1])

    @staticmethod
    def _compress_coords(arr:list) -> None:
        unique_coords = sorted(set(arr)) # O(n log n)
        coord_dict = {coord: i for i, coord in enumerate(unique_coords)} # O(n)
        compressed_coords = [coord_dict[coord] for coord in arr]
        
        for i in range(len(arr)):
            arr[i] = compressed_coords[i]

    @staticmethod
    def _get_number_of_clusters(carr:list) -> int:
        if not carr: return 0
        return max(carr) + 1
    
    @staticmethod
    def _get_size_of_clusters(carr:list) -> list:
        n_cluster = YOLOCluster._get_number_of_clusters(carr)
        ret = [0] * n_cluster
        for i in range(len(carr)):
            ret[carr[i]] += 1
        return ret
    
    @staticmethod
    def _get_dealer_index(yresult) -> None:
        carr = yresult.carr
        n = len(carr)
        if n == 0:
            ret = -1
        
        elif n == 1:
            ret = 0
        
        else:
            n_clusters = YOLOCluster._get_number_of_clusters(carr)
            size_of_clusters = YOLOCluster._get_size_of_clusters(carr)

            avgy = [0] * n_clusters

            mxyarr = yresult.mxyarr

            for i in range(n):
                y = mxyarr[i][1]
                clus = carr[i]
                avgy[clus] += y
            
            for i in range(n_clusters):
                avgy[i] /= size_of_clusters[i]

            ret = np.argmin(avgy)
        
        yresult.dealeri = ret

    

    @staticmethod
    def _get_carr(yresult) -> None:
        xyxycpu = yresult.boxes.xyxy.cpu()
        height = yresult.orig_shape[0]
        width = yresult.orig_shape[1]

        n = len(xyxycpu)

        if n == 0:
            ret = []
            yresult.carr = ret
        
        elif n == 1:
            ret = [0]
            yresult.carr = ret
        
        else:
            YOLOCluster._get_mxyarr(yresult)
            
            standard = min(width, height) * YOLOCluster.C_RATIO

            # -1 means it is not belong to any cluster yet
            ret = [-1] * n
            uniqueid = 0

            for i in range(n):

                # if i has no cluster, assign unique cluster
                if ret[i] == -1:
                    ret[i] = uniqueid
                    uniqueid += 1

                for j in range(i+1, n):
                    distance = YOLOCluster._get_distance_pt1_pt2(yresult.mxyarr[i], yresult.mxyarr[j])

                    # if i and j should be in the same cluster
                    if distance < standard:
                        if ret[j] != -1:
                            ret[i] = ret[j]
                        else:
                            ret[j] = ret[i]
                
            YOLOCluster._compress_coords(ret)

            # ret e.g. [0 0 0 1 0 1 1 2 2 0 1 2 3]
            yresult.carr = ret

    @staticmethod
    def _make_dealer_first(yresult):
        carr = yresult.carr
        YOLOCluster._get_dealer_index(yresult)
        dealer_index = yresult.dealeri
        if dealer_index > 0:
            for i in range(len(carr)):
                if carr[i] == 0:
                    carr[i] = dealer_index
                elif carr[i] == dealer_index:
                    carr[i] = 0

    @staticmethod
    def _get_class_of_box(box:ultralytics.engine.results.Boxes) -> int:
        # type name is "Boxes", but argument is single
        return int(box.cls.cpu()[0])
    
    @staticmethod
    def _get_count_dict(bagg:tuple) -> dict:
        ret = dict()
        for i in bagg:
            if i not in ret:
                ret[i] = 1
            else:
                ret[i] += 1
        
        #e.g. {"KH": 2, "9S": 1, "6D": 3}
        return ret

    @staticmethod
    def _get_plain_result(yresult) -> None:
        boxes = yresult.boxes

        class_names = yresult.names

        nclusters = YOLOCluster._get_number_of_clusters(yresult.carr)
        ret = [None] * nclusters
        for i in range(nclusters):
            ret[i] = tuple()

        n = len(yresult.carr)

        for i in range(n):
            box = boxes[i]
            cls = YOLOCluster._get_class_of_box(box)
            nameofclass = class_names[cls]
            cluster_idx = yresult.carr[i]
            ret[cluster_idx] += (nameofclass,)
        
        # ret e.g. (("KH", "KH"), ("9S",))
        yresult.parr = ret
    
    @staticmethod
    def _match_plain_result(plain_result:list) -> None:
        n = len(plain_result)

        for i in range(n):
            clust = plain_result[i]
            count_dict = YOLOCluster._get_count_dict(clust)
            newtuple = tuple()
            for key in count_dict:
                count = count_dict[key]
                card_count = (count + 1) // 2
                newtuple += (key,) * card_count
            plain_result[i] = newtuple

    @staticmethod
    def _predictc(yresult) -> None:
        YOLOCluster._get_carr(yresult)
        YOLOCluster._make_dealer_first(yresult)

        YOLOCluster._get_plain_result(yresult)
        parr_clone = yresult.parr[:]

        YOLOCluster._match_plain_result(parr_clone)
        yresult.mparr = parr_clone
    
    @staticmethod
    def plotc(yresult) -> np.ndarray:
        im = yresult.orig_img
        carr = yresult.carr

        annotator = Annotator(im)
        box_index = 0
        for box in yresult.boxes:
            b = box.xyxy[0]
            c = box.cls
            class_index = int(c)
            class_name = yresult.names[class_index]
            s_conf = "%.2f"%float(box.conf[0].cpu())
            cluster = carr[box_index]
            label = "%s %s %s"%(class_name, s_conf, cluster)
            annotator.box_label(b, label)
            box_index += 1

        im = annotator.result()
        return im
