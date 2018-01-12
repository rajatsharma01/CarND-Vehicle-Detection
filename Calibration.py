import numpy as np
import glob
import cv2

# This class will calibarate camera and uses camera matrix to undistort images
class Calibrate():
    def __init__(self):
        image_points = [] # 2d points in image fram
        obj_points = [] # 3d points in real world
        # Internal corner points dimensions in chessboard images
        ny, nx = (6, 9)

        # create 3d object points as all internal corner points in chesboard
        objp = np.zeros((ny*nx, 3), np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

        # Read all calibration images and find corners
        cal_images = sorted(glob.glob('./camera_cal/calibration*.jpg'))
        for imfile in cal_images:
            image = cv2.imread(imfile)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            if ret == True:
                image_points.append(corners)
                obj_points.append(objp)

        # Calibrate camera with image and object points
        test_image = cv2.imread('./camera_cal/calibration1.jpg')

        # Note that calibrateCamera accepts image_size as (width, height)
        ret, self.cam_mtx, self.dist_coeff, rvecs, tvecs = \
        cv2.calibrateCamera(obj_points, image_points, test_image.shape[1::-1], None, None)

    # Return an undistorted img for input img
    def undistort(self, img):
        return cv2.undistort(img, self.cam_mtx, self.dist_coeff, None, self.cam_mtx)
