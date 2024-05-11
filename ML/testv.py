BEST = r"C:\Blackjack-copilot\ML\yolo\train_workspace\runs\detect\train\weights\best.pt"
VIDEO_PATH = r"C:\Blackjack-copilot\ML\yolo\train_workspace\extern_test_videos\blackjack.mp4"

import cv2
import yolo
import cluster

def visualize_video():
    model=yolo.get_model(BEST)
    results = model.predict(VIDEO_PATH, stream=True, imgsz=(720,1280))
    for yresult in results:
        vis = cluster.visualize_by_yresult(yresult)
        print(cluster.get_final_result_by_yresult(yresult))
        cv2.imshow("winname", vis)
        if cv2.waitKey(1) > 0: break

visualize_video()

#TODO:Fix unknown error
"""
Traceback (most recent call last):
  File "c:\Blackjack-copilot\ML\testv.py", line 17, in <module>
    visualize_video()
  File "c:\Blackjack-copilot\ML\testv.py", line 12, in visualize_video
    vis = cluster.visualize_by_yresult(yresult)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "c:\Blackjack-copilot\ML\cluster.py", line 262, in visualize_by_yresult
    cresult = predict(yresult)
              ^^^^^^^^^^^^^^^^
  File "c:\Blackjack-copilot\ML\cluster.py", line 130, in predict
    matchedxyarr = get_matched_xy_arr(yresult)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "c:\Blackjack-copilot\ML\cluster.py", line 103, in get_matched_xy_arr
    pairs = get_pairs_with_yresult(yresult)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "c:\Blackjack-copilot\ML\cluster.py", line 94, in get_pairs_with_yresult
    return get_pairs_from_matcharr(match_cards(yresult))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "c:\Blackjack-copilot\ML\cluster.py", line 85, in get_pairs_from_matcharr
    j = matcharr.index(cls, i+1)
        ^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: 0 is not in list"""
