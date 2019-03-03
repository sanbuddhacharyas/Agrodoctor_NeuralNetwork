import numpy as np
import cv2
import h5py
from camera_calibrate import StereoCalibration

cal = StereoCalibration('')

#data = cal.camera_model

numBoards = 27  #how many boards would you like to find
board_w = 9
board_h = 7

board_sz = (9,7)
board_n = board_w*board_h

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all the images.
object_points = [] # 3d point in real world space
imagePoints1 = [] # 2d points in image plane.
imagePoints2 = [] # 2d points in image plane.

corners1 = []
corners2 = []

#obj = []
#for j in range(0,board_n):
    #obj.append(np.(j/board_w, j%board_w, 0.0))
obj = np.zeros((7*9,3), np.float32)
obj[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)
img1 = cv2.imread("left/Left0.jpg")
height, width, depth  = img1.shape

"""

vidStreamL = cv2.VideoCapture(1)  # index of your camera\
vidStreamL.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
vidStreamL.set(cv2.CAP_PROP_FRAME_HEIGHT,600)
vidStreamR = cv2.VideoCapture(2)  # index of your camera
vidStreamR.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
vidStreamR.set(cv2.CAP_PROP_FRAME_HEIGHT,600)
success = 0
k = 0
found1 = False
found2 = False






rectify_scale = 0
R1 = np.zeros(shape=(3,3))
R2 = np.zeros(shape=(3,3))
P1 = np.zeros(shape=(3,3))
P2 = np.zeros(shape=(3,3))
#(roi1, roi2) = cv2.cv.StereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,(width, height), R, T, R1, R2, P1, P2, Q=None, flags=cv2.cv.CV_CALIB_ZERO_DISPARITY, alpha=-1, newImageSize=(0, 0))
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(data['M1'], data['dist1'], data['M2'], data['dist2'], (width, height),data['R'],data['T'],R1,R2,P1,P2,Q=None, alpha=-1, newImageSize=(0, 0))

stereo_par={

"R1":R1,
"R2":R2,
"P1":P1,
"P2":P2,
"Q":Q,
"roi1":roi1,
"roi2":roi2
}

#cal.write_data(data , stero_parameter)

#stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,(width, height), R, T)


print ("Done Rectification\n")
print ("Applying Undistort\n")



map1x, map1y = cv2.initUndistortRectifyMap(data['M1'], data['dist1'], stereo_par["R1"], stereo_par["P1"], (width, height), cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(data['M2'], data['dist2'], stereo_par["R2"], stereo_par["P2"], (width, height), cv2.CV_32FC1)

map ={
    "map1x":map1x,
    "map1y":map1y,
    "map2x":map2x,
    "map2y":map2y
}
#cal.write_data(data,stereo_par,map)
"""
data , stereo ,map = cal.read_data()
print ("Undistort complete\n")


success =0
while(True):
    stringL = "left/Left"+str(success)+".jpg"
    stringR = "right/Right"+str(success)+".jpg"
    img1 = cv2.imread(stringL)
    img2 = cv2.imread(stringR)


    imgU1 = cv2.remap(img1,map["map1x"],map["map1y"], cv2.INTER_LANCZOS4)
    imgU2 = cv2.remap(img2, map["map2x"], map["map2y"], cv2.INTER_LANCZOS4)

    found1 ,corners1 = cv2.findChessboardCorners(imgU1,board_sz)
    found2 ,corners2 = cv2.findChessboardCorners(imgU2,board_sz)
    
    f1 ,c1 = cv2.findChessboardCorners(img1,board_sz)
    f2 ,c2 = cv2.findChessboardCorners(img2,board_sz)
    if(found1 !=0 and found2!=0):
        if corners1[0][0][0] > corners1[62][0][0]:
                corners1 = np.rot90(corners1,2).reshape(63,1,2)
                corners1 = np.array(corners1)

        if corners2[0][0][0] > corners2[62][0][0]:
                corners2 = np.rot90(corners2,2).reshape(63,1,2)
                corners2 = np.array(corners2)
    if(f1 !=0 and f2!=0):
        if c1[0][0][0] > c1[62][0][0]:
            c1 = np.rot90(c1,2).reshape(63,1,2)
            c1 = np.array(c1)

        if c2[0][0][0] > c2[62][0][0]:
            c2 = np.rot90(c2,2).reshape(63,1,2)
            c2 = np.array(c2)

        diff = corners1[0][0][1] - corners2[0][0][1]
        diff1 = c1[0][0][1] - c2[0][0][1]     
        print("without->"+str(diff)+"with->"+str(diff1))
        cv2.drawChessboardCorners(imgU1,board_sz,corners1,found1)
        cv2.drawChessboardCorners(imgU2,board_sz,corners2,found2)

        cv2.drawChessboardCorners(img1,board_sz,c1,f1)
        cv2.drawChessboardCorners(img2,board_sz,c2,f2)

    cv2.imshow("image1L", imgU1)
    cv2.imshow("image2R", imgU2)
    cv2.imshow("image1", img1)
    cv2.imshow("image2", img2)
    k = cv2.waitKey(5)
    if(k==27):
        break

    if(k == 32):
        success+=1
    
    

