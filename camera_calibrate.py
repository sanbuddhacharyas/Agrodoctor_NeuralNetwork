import numpy as np
import cv2
import glob
import argparse
import h5py


class StereoCalibration(object):
    def __init__(self, filepath):
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
                             cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((9*7, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

        self.cal_path = filepath
        #self.read_images(self.cal_path)

    def read_images(self, cal_path):
        images_right = glob.glob('right/*.jpg')
        images_left = glob.glob('left/*.jpg')
        images_left.sort()
        images_right.sort()

        for i, fname in enumerate(images_right):
            img_l = cv2.imread(images_left[i])
            img_r = cv2.imread(images_right[i])

            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (9, 7), None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (9, 7), None)
            print(corners_l)
            if corners_l[0][0][0] > corners_l[62][0][0]:
                corners_l = np.rot90(corners_l,2).reshape(63,1,2)
                corners_l = np.array(corners_l)
                print(corners_l)

            if corners_r[0][0][0] > corners_r[62][0][0]:
                corners_r = np.rot90(corners_r,2).reshape(63,1,2)
                corners_r = np.array(corners_r)
                print(corners_r)

            # If found, add object points, image points (after refining them)
            self.objpoints.append(self.objp)

            if ret_l is True:
                rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), self.criteria)
                self.imgpoints_l.append(corners_l)

                # Draw and display the corners
                ret_l = cv2.drawChessboardCorners(img_l, (9, 7),corners_l, ret_l)
                cv2.imshow(images_left[i], img_l)
                cv2.waitKey(500)

            if ret_r is True:
                rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11),(-1, -1), self.criteria)
                self.imgpoints_r.append(corners_r)

                # Draw and display the corners
                ret_r = cv2.drawChessboardCorners(img_r, (9, 7),corners_r, ret_r)
                cv2.imshow(images_right[i], img_r)
                cv2.waitKey(500)
            img_shape = gray_l.shape[::-1]

        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, img_shape, None, None)
        rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, img_shape, None, None)

        self.camera_model = self.stereo_calibrate(img_shape)

    def stereo_calibrate(self, dims):
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5

        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l,
            self.imgpoints_r, self.M1, self.d1, self.M2,
            self.d2, dims,
            criteria=stereocalib_criteria, flags=flags)

        print('Intrinsic_mtx_1', M1)
        print('dist_1', d1)
        print('Intrinsic_mtx_2', M2)
        print('dist_2', d2)
        print('R', R)
        print('T', T)
        print('E', E)
        print('F', F)

        # for i in range(len(self.r1)):
        #     print("--- pose[", i+1, "] ---")
        #     self.ext1, _ = cv2.Rodrigues(self.r1[i])
        #     self.ext2, _ = cv2.Rodrigues(self.r2[i])
        #     print('Ext1', self.ext1)
        #     print('Ext2', self.ext2)

        print('')

        camera_model = dict([('M1', M1), ('M2', M2), ('dist1', d1),
                            ('dist2', d2), ('rvecs1', self.r1),
                            ('rvecs2', self.r2), ('R', R), ('T', T),
                            ('E', E), ('F', F)])

        cv2.destroyAllWindows()
        return camera_model

    def write_data(self,Calibration_parameter, Stereo_parameters ,map):
        with h5py.File("Calibration.h5py","w") as hdf:
            Camera_p = hdf.create_group("Camera_parameter")
            Stereo_p = hdf.create_group("Stereo_parameters")
            map_p = hdf.create_group("map")
            
            Camera_p.create_dataset("M1",data = Calibration_parameter["M1"])
            Camera_p.create_dataset("dist1",data = Calibration_parameter["dist1"] )
            Camera_p.create_dataset("M2",data = Calibration_parameter["M2"] )
            Camera_p.create_dataset("dist2",data = Calibration_parameter["dist2"] )
            Camera_p.create_dataset("R",data = Calibration_parameter["R"] )
            Camera_p.create_dataset("T",data = Calibration_parameter["T"] )
            Camera_p.create_dataset("E",data = Calibration_parameter["E"] )
            Camera_p.create_dataset("F",data = Calibration_parameter["F"] )

            Stereo_p.create_dataset("R1",data = Stereo_parameters["R1"])
            Stereo_p.create_dataset("R2",data = Stereo_parameters["R2"] )
            Stereo_p.create_dataset("P1",data = Stereo_parameters["P1"] )
            Stereo_p.create_dataset("P2",data = Stereo_parameters["P2"] )
            Stereo_p.create_dataset("Q",data = Stereo_parameters["Q"] )
            Stereo_p.create_dataset("roi1",data = Stereo_parameters["roi1"] )
            Stereo_p.create_dataset("roi2",data = Stereo_parameters["roi2"] )

            map_p.create_dataset("map1x",data=map["map1x"])
            map_p.create_dataset("map1y",data=map["map1y"])
            map_p.create_dataset("map2x",data=map["map2x"])
            map_p.create_dataset("map2y",data=map["map2y"])

           

            #data.create_dataset("Stereo_parameters",data= Stereo_parameters)

            

            

    def read_data(self):
        with h5py.File("Calibration.h5py","r") as hdf:
            hdf.keys()
            d ={
            "M1":np.array(hdf["Camera_parameter/M1"]),
            "dist1": np.array(hdf["Camera_parameter/dist1"]),
            "M2": np.array(hdf["Camera_parameter/M2"]),
            "dist2": np.array(hdf["Camera_parameter/dist2"]),
            "R" : np.array(hdf["Camera_parameter/R"]),
            "T" : np.array(hdf["Camera_parameter/T"]),
            "E" : np.array(hdf["Camera_parameter/E"]),
            "F" : np.array(hdf["Camera_parameter/F"])
            }
            stereo={
            "R1": np.array(hdf["Stereo_parameters/R1"]),
            "R2" : np.array(hdf["Stereo_parameters/R2"]),
            "P1" : np.array(hdf["Stereo_parameters/P1"]),
            "P2" : np.array(hdf["Stereo_parameters/P2"]),
            "Q" :  np.array(hdf["Stereo_parameters/Q"]),
            "roi1":np.array(hdf["Stereo_parameters/roi1"]),
            "roi2":np.array(hdf["Stereo_parameters/roi2"])
            }
        
            map ={
            "map1x":np.array(hdf["map/map1x"]),
            "map1y":np.array(hdf["map/map1y"]),
            "map2x":np.array(hdf["map/map2x"]),
            "map2y":np.array(hdf["map/map2y"])
            }
        
        return d, stereo,map
         


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='String Filepath')
    args = parser.parse_args()
    cal_data = StereoCalibration(args.filepath)