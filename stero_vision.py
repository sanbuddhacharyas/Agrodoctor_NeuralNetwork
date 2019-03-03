import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from yolo import stereo_vision
stv = stereo_vision(0.5, 0.6)


capL = cv2.VideoCapture(2)
capL.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
capR = cv2.VideoCapture(1)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)


while True:
    #frameL = cv2.imread('bottle/bottle0.png')
    #frameR = cv2.imread('bottle/bottle1.png')
    retL, frameL  = capL.read(cv2.IMREAD_COLOR)
    print(frameL.shape)
    retR, frameR  = capR.read(cv2.IMREAD_COLOR)
    #distance_measured = stv.object_finding(frameL , frameR ,"bottle")
   # print("distance_measured"+str(distance_measured))
    
    cv2.imshow('Left_camera',frameL)
    cv2.imshow('Righr_camera',frameR)
       
    K = cv2.waitKey(1)
    if K == 27:
        capL.release()
        capR.release()
        cv2.destroyAllWindows()
        break       
        

            
