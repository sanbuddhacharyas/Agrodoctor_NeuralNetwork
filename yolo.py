import cv2
import numpy as np
from camera_calibrate import StereoCalibration
from darkflow.net.build import TFNet
import math
class stereo_vision:

    def __init__(self,threshold,gpu):
        options ={
        'model' : 'cfg/yolo.cfg',
        'load' : 'bin/yolo.weights',
        'threshold': threshold,
        'gpu' : gpu
        }
        cal = StereoCalibration('')
        self.data, self.stereo ,self.map = cal.read_data()
        self.tfnet = TFNet(options)
        
    def distance_measurement(self, resL ,resR, frameL, frameR,rang):
        count =0
        subtracted_mean_pre = 1000
        differeceL = abs(resL['bottomright']['x'] - resL['topleft']['x'])
        differece =  abs(resL['bottomright']['y'] - resL['topleft']['y'])
        gray_croped_L = cv2.cvtColor(frameL, cv2.COLOR_RGB2GRAY)
        gray_croped_L = gray_croped_L[resL['topleft']['y'] + int(0.1 * differece) : resL['bottomright']['y'] -int(0.1 * differece), resL['topleft']['x'] + int(0.1 * differeceL) : resL['bottomright']['x']- int( 0.1 * differeceL)]
        gray_R = cv2.cvtColor(frameR, cv2.COLOR_RGB2GRAY)
        cv2.imshow("gray_croped",gray_croped_L)
        cv2.waitKey(0)
        for i in range(rang):
            gray_croped_R = gray_R[resL['topleft']['y']:resL['bottomright']['y'],resR['topleft']['x']+i-25: resR['topleft']['x']+i-25+differeceL]
            cv2.imshow("CR",gray_croped_R)
            subtracted_image = np.abs(np.subtract(gray_croped_R ,gray_croped_L))
            subtracted_mean =  np.sum(subtracted_image.flatten(),axis=0)
            #print("subtracted_mean"+str(subtracted_mean))
            
            if subtracted_mean < subtracted_mean_pre:
                count = i
                subtracted_mean_pre = subtracted_mean

        disparity1 = abs((resR['topleft']['x']+count-15) - resL['topleft']['x'] )
        disparity = abs(resL["middleL"]-resR["middleR"])
        x = frameL.shape[0]
        print("disparity1:"+str(disparity1)+"dis:"+str(disparity))

        distance = (13*x)/(math.tan(math.radians(30))*disparity)
        distance1 = (13*x)/(math.tan(math.radians(30))*disparity1)
        distance = distance - 3.3
        distance1 =distance1 - 1
        
        return distance , distance1
        
                        



    def object_finding(self, frameL, frameR, search):
        Left = []
        Right = []
        imgU1 = cv2.remap(frameL, self.map["map1x"], self.map["map1y"],cv2.INTER_LANCZOS4)
        imgU2 = cv2.remap(frameR, self.map["map2x"], self.map["map2y"],cv2.INTER_LANCZOS4)
        resultL = self.tfnet.return_predict(imgU1)
        resultR = self.tfnet.return_predict(imgU2)
        distance = {}

        if not resultL == []:
            for res in resultL:
                if res['label'] == search:
                    res.update({"AreaL" : (res['bottomright']['x']- res['topleft']['x']) * (res['bottomright']['y'] - res['topleft']['y']) })
                    res.update({"middleL": res['topleft']['x'] + (res['bottomright']['x'] - res['topleft']['x'])/2 })
                    Left.append(res)
                

        if not resultR == []:
            for res in resultR:
                if res['label'] == search:
                    res.update({"AreaR" : (res['bottomright']['x']- res['topleft']['x']) * (res['bottomright']['y'] - res['topleft']['y']) })
                    res.update({"middleR": res['topleft']['x'] + (res['bottomright']['x'] - res['topleft']['x'])/2 })
                    Right.append(res)
        m=0
        for left in Left:
            for right in Right:
                if abs(left["AreaL"] - right["AreaR"])<1100:
                    distance.update({"dis"+str(m) : self.distance_measurement(left , right ,imgU1 ,imgU2 ,50)})
                    m += 1
                
                
                print("Left->"+str(left) + " Right->"+str(right))



        
        return distance

                    




    
               