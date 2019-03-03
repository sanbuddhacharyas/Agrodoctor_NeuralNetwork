import cv2 as cv
import numpy as np

PATTEN_SIZE = (9 , 7)
class calibrate:

    
    def findpoints(self, images):
       
        img_points=[]

        stop_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,30, 0.001)

        for (image,color_image) in images:
            found, corners = cv.findChessboardCorners(image, PATTEN_SIZE, None)
            if found:
                self.obj_points.append(self.obj)
                cv.cornerSubPix(image, corners,(11, 11),(-1, -1),stop_criteria)
                img_points.append(corners)

            cv.drawChessboardCorners(color_image, PATTEN_SIZE,corners, found)
            

        return img_points

    def stero_Calibration(self , imagesL, imagesR):
        img_points_L = findpoints(imagesL)

        stereocalib_criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 100, 1e-5)
        stereocalib_flags = cv.CALIB_FIX_ASPECT_RATIO | cv.CALIB_ZERO_TANGENT_DIST | cv.CALIB_SAME_FOCAL_LENGTH | cv.CALIB_RATIONAL_MODEL | cv.CALIB_FIX_K3 | cv.CALIB_FIX_K4 | cv.CALIB_FIX_K5
        stereocalib_retval, CM1, DC1, CM2, DC2, R, T, E, F = cv.stereoCalibrate(self.obj_points,img_left_points,img_right_points,image_size,criteria = stereocalib_criteria, flags = stereocalib_flags)
    

        