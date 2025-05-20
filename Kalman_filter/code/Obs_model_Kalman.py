import scipy.io as sio
import numpy as np
from os.path import dirname, join as pjoin
import scipy.io as sio
from scipy.spatial.transform import Rotation as R
import cv2
from matplotlib import pyplot as plt 
from scipy.signal import savgol_filter

#class Observation_model
class Observation_model():
    def __init__(self):
        ##This function sets all required fields such as roation and position
        ##IMU frame is algined with drone's body frame
        ##For postion and rotation notations used for camera, IMU and world are c, r, w respectively.
        
        ##set camera matrices, intrinsic extrinsic
        self.camera_mat = np.array([
                [314.1779, 0.0,     199.4848],
                [0.0,      314.2218, 113.7838],
                [0.0,      0.0,      1.0]
            ])

        self.dist_coeffs = np.array([-0.438607, 0.248625, 0.00072, -0.000476, -0.0911])

        ## set Camera to IMU transformation
        ## Rotation from camera frame to IMU/Robot frame: rotation about the camera’s z-axis of
        ## 𝜋/4 rad and about the x-axis of 𝜋 rad.
        R_z = self.rotate(axis = 'z', theta = -np.pi/4)
        R_x = self.rotate(axis = 'x', theta = np.pi)
        self.R_cr =   R_z @ R_x
        
        ##Translation from drone to camera
        self.P_cr = np.array([-0.04, 0.0, -0.03])
        self.T_cr = np.vstack([np.hstack([self.R_cr.reshape(3,3), self.P_cr.reshape(3,1)]),
                          np.hstack([0,0,0.0,1] )]) 
        
        ##tag data
        self.tag_space = 0.152
        self.tag_space_mid = 0.178 
        self.tag_ids = np.array([
                             [  0,  12,  24,  36,  48,  60,  72,  84,  96],
                             [  1,  13,  25,  37,  49,  61,  73,  85,  97],
                             [  2,  14,  26,  38,  50,  62,  74,  86,  98],
                             [  3,  15,  27,  39,  51,  63,  75,  87,  99],
                             [  4,  16,  28,  40,  52,  64,  76,  88, 100],
                             [  5,  17,  29,  41,  53,  65,  77,  89, 101],
                             [  6,  18,  30,  42,  54,  66,  78,  90, 102],
                             [  7,  19,  31,  43,  55,  67,  79,  91, 103],
                             [  8,  20,  32,  44,  56,  68,  80,  92, 104],
                             [  9,  21,  33,  45,  57,  69,  81,  93, 105],
                             [ 10,  22,  34,  46,  58,  70,  82,  94, 106],
                             [ 11,  23,  35,  47,  59,  71,  83,  95, 107]
                            ], dtype=int)
        self.P_cw_last = np.array([])
        self.R_cw_last = np.array([])
        
    def rotate(self, axis, theta):
        ## calculate rotation around the axis
        ## theta is in radians
        
        if axis == 'z':
           R =  np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta),  np.cos(theta), 0],
                [0.0,              0,             1]
                ])
        elif axis == 'y':
            R = np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0.0,             1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
                ])
        elif axis == 'x':
            R = np.array([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0.0, np.sin(theta),  np.cos(theta)]
                ])
        else:
            print("wrong axis is passed")
            return None
            
        return R
      
    def cal_tag_corners(self, idx):
        ## this calculates april tag's corner in world frame 
        idx = np.argwhere(self.tag_ids == idx)
        if len(idx) == 0:
            print("ID not found!",  idx)
            return []
        row, col = idx[0]
    
        x = row * (2 * self.tag_space)
        gaps = int(col >= 3) + int(col >= 6)
        y = self.tag_space * (2 * col - gaps) + self.tag_space_mid * gaps    
        p1 = [x + self.tag_space, y              , 0.0]
        p2 = [x + self.tag_space, y + self.tag_space, 0.0]
        p3 = [x                , y + self.tag_space, 0.0]
        p4 = [x                , y              , 0.0]
        
        return np.array([p1, p2, p3, p4], dtype = "float32")
        
    def measure_drone_pose(self, data):
         
        self.K = 10
        T_cw = self.cal_camera_pose(data)
        if len(T_cw) == 0:
            return np.array([])
        T_wr = np.linalg.pinv(T_cw) @ self.T_cr 
        position = T_wr[0:3, 3 ]
        R_wr = T_wr[0:3,0:3]
        rot = R.from_matrix(R_wr)
        phi, theta, psi = rot.as_euler('xyz', degrees=False)
        orientation = np.array([phi, theta, psi])  
        pose = np.hstack([position , orientation]).reshape(6,1)       
                   
        return pose
      
    def cal_camera_pose(self, data):
        if len(data) == None:
            return []
        points = ['p1', 'p2', 'p3', 'p4']
        world_corners = []
        camera_corners = []   
        id = np.atleast_1d(data.get('id', []))      

        ##id could be int or np array or block array
        ## did error handling here
        if isinstance(id, int):            
            ID_list = [id]
        else:
            ID_list = id.flatten()
                        
        if len(ID_list)==0:
            return []

        for i, ID in enumerate(ID_list):  
            corners = self.cal_tag_corners(ID)
            if len(corners) == 0:
                return []
               
            
            #get corners locations in image frmae
            pixel_list = []
            for point in points: 
                ##exception handling
                try:
                    px = data[point][:,i] 
                    #print("sh", px.shape[0])
                    if px.shape[0] == 2:  
                        #print(px)
                        pixel_list.append(px)
                    elif px.shape[0] == 1 :
                        #print(px.T)
                        list1 = [pt for pt in px[0].T]
                        pixel_list.extend(list1)
                        
                except:
                    print("could not find pixel value for ID") 
                    continue

            pix_array = np.array(pixel_list)           
            if corners.shape == (4,3) and pix_array.shape ==(4,2):                
                camera_corners.append(pixel_list[0:4])
                world_corners.append(corners ) 
            
         
        world_corners = np.array(world_corners, dtype = "float32").reshape(-1, 3)        
        camera_corners = np.array(camera_corners, dtype = "float32").reshape(-1,2)

        ## exception handling
        if (world_corners.shape[0]==0 or camera_corners.shape[0]== 0 ):
            return []
        

        ## PNP ransac is used to calculate world to camera transformation is calculated here
        success, R_vec, P_cw, inliers = cv2.solvePnPRansac(
        world_corners,          # 3D object points (Nx3)
        camera_corners,         # 2D image points (Nx2)
        self.camera_mat,        # Camera intrinsic matrix (3x3)
        self.dist_coeffs,       # Distortion coefficients (5x1 or 1x5)
        flags=cv2.SOLVEPNP_ITERATIVE,  # EPNP is good with many points, or use ITERATIVE
        reprojectionError=10.0,    # Max allowed pixel error
        confidence=0.90,          # Confidence level
        iterationsCount=10000       # RANSAC iterations
        )

        if success != True:
            return []      
        #print(success)
        ## calculate apply LM pnp for refining
        R_vec, P_cw = cv2.solvePnPRefineLM(
            world_corners[inliers], camera_corners[inliers],
            self.camera_mat, self.dist_coeffs, R_vec, P_cw
        )
        ## world to camera transformation is calculated here
        R_cw,_ = cv2.Rodrigues(R_vec)        
        T_cw = np.vstack([np.hstack([R_cw,P_cw.reshape(3,1)] ),
                     np.hstack([0,0,0.0,1])])
        
        return T_cw 
    

    def get_covar_mat(self):
        R =  np.array([
        [0.00275182, 0.0071382 , 0.0006617 ,  0.00105378, 0.00172351, 0.00328077],
        [0.0071382 , 0.08327198, 0.00054009, 0.00881066, 0.00437954, 0.03826406],
        [0.0006617 , 0.00054009, 0.02187695, -0.0021889 , -0.00882777, 0.00135139],
        [0.00105378, 0.00881066, -0.0021889 , 0.00515638, 0.00237747, 0.00530565],
        [0.00172351, 0.00437954, -0.00882777, 0.00237747, 0.00646523, 0.00210105],
        [0.00328077, 0.03826406, 0.00135139, 0.00530565, 0.00210105, 0.02192298]
        ])
        return R