# Main for start threads
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import threading
from Functions import * 
#from queue import Process, Queue
import multiprocessing as mps
# define a video capture object

if __name__ == "__main__":
    print("Number of cpu : ", mps.cpu_count())
    Frame_share = mps.Queue()
    stop_threads_share = mps.Queue()
    stop_threads_share.put(False)

    try:
        mps.set_start_method('spawn')
    except RuntimeError:
        pass
    ######
    # In my pc the canon camera is the number 2;
    Camera_number = 2
    ######
    T_capture = mps.Process(target=thread_Capture, args=(Camera_number,Frame_share,stop_threads_share,))  
      
    T_Sobel = mps.Process(target=thread_Sobel, args=(Frame_share,stop_threads_share,))

    T_capture.start()
    T_Sobel.start()
    #T_Sobel.join()
    

    