# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
from decimal import Decimal
import cv2
import dlib
import math
import os

'''
# @function:    人脸检测、关键点检测+人头姿态估计
# @author:      TheDetial
# @date:        
# @last_edit:   

'''

# 获取相机参数
def camParm(parm_path):
    # 相机内参
    K = [0.0] * 9
    # 畸变矩阵
    # D = [0.0]*5
    D = []

    fopen = open(parm_path)  # parm_list
    lines = fopen.readlines()
    for line in lines:
        line = line.replace('\n', '')  # remove 原始str中的末尾 '\n',否则末尾添加新str元素会换行
        line = line.split(',')
        K[0] = float(line[0])  # fx
        K[4] = float(line[1])  # fy
        K[2] = float(line[2])  # cx
        K[5] = float(line[3])  # cy
        K[8] = 0.0
        # K1,K2,P1,P2,K3
        D.append(float(line[4]))
        D.append(float(line[5]))
        D.append(float(line[6]))
        D.append(float(line[7]))
        D.append(float(line[8]))

    cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
    dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)
    return cam_matrix, dist_coeffs

# 人脸关键点检测+头部姿态估计
class faceLandMarksDect:

    def __init__(self, mode_path, parm_path, data_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor_path = mode_path
        self.cam_matrix, self.dist_coeffs = camParm(parm_path)
        self.data_path = data_path
        self.model_points_68 = np.array([  # open_face_wild_face_3D_model_68_landmarks_(x,y,z)
            [-73.393523, -29.801432, -47.667532],
            [-72.775014, -10.949766, -45.909403],
            [-70.533638, 7.929818, -44.84258],
            [-66.850058, 26.07428, -43.141114],
            [-59.790187, 42.56439, -38.635298],
            [-48.368973, 56.48108, -30.750622],
            [-34.121101, 67.246992, -18.456453],
            [-17.875411, 75.056892, -3.609035],
            [0.098749, 77.061286, 0.881698],
            [17.477031, 74.758448, -5.181201],
            [32.648966, 66.929021, -19.176563],
            [46.372358, 56.311389, -30.77057],
            [57.34348, 42.419126, -37.628629],
            [64.388482, 25.45588, -40.886309],
            [68.212038, 6.990805, -42.281449],
            [70.486405, -11.666193, -44.142567],
            [71.375822, -30.365191, -47.140426],
            [-61.119406, -49.361602, -14.254422],
            [-51.287588, -58.769795, -7.268147],
            [-37.8048, -61.996155, -0.442051],
            [-24.022754, -61.033399, 6.606501],
            [-11.635713, -56.686759, 11.967398],
            [12.056636, -57.391033, 12.051204],
            [25.106256, -61.902186, 7.315098],
            [38.338588, -62.777713, 1.022953],
            [51.191007, -59.302347, -5.349435],
            [60.053851, -50.190255, -11.615746],
            [0.65394, -42.19379, 13.380835],
            [0.804809, -30.993721, 21.150853],
            [0.992204, -19.944596, 29.284036],
            [1.226783, -8.414541, 36.94806],
            [-14.772472, 2.598255, 20.132003],
            [-7.180239, 4.751589, 23.536684],
            [0.55592, 6.5629, 25.944448],
            [8.272499, 4.661005, 23.695741],
            [15.214351, 2.643046, 20.858157],
            [-46.04729, -37.471411, -7.037989],
            [-37.674688, -42.73051, -3.021217],
            [-27.883856, -42.711517, -1.353629],
            [-19.648268, -36.754742, 0.111088],
            [-28.272965, -35.134493, 0.147273],
            [-38.082418, -34.919043, -1.476612],
            [19.265868, -37.032306, 0.665746],
            [27.894191, -43.342445, -0.24766],
            [37.437529, -43.110822, -1.696435],
            [45.170805, -38.086515, -4.894163],
            [38.196454, -35.532024, -0.282961],
            [28.764989, -35.484289, 1.172675],
            [-28.916267, 28.612716, 2.24031],
            [-17.533194, 22.172187, 15.934335],
            [-6.68459, 19.029051, 22.611355],
            [0.381001, 20.721118, 23.748437],
            [8.375443, 19.03546, 22.721995],
            [18.876618, 22.394109, 15.610679],
            [28.794412, 28.079924, 3.217393],
            [19.057574, 36.298248, 14.987997],
            [8.956375, 39.634575, 22.554245],
            [0.381549, 40.395647, 23.591626],
            [-7.428895, 39.836405, 22.406106],
            [-18.160634, 36.677899, 15.121907],
            [-24.37749, 28.677771, 4.785684],
            [-6.897633, 25.475976, 20.893742],
            [0.340663, 26.014269, 22.220479],
            [8.444722, 25.326198, 21.02552],
            [24.474473, 28.323008, 5.712776],
            [8.449166, 30.596216, 20.671489],
            [0.205322, 31.408738, 21.90367],
            [-7.198266, 30.844876, 20.328022]], dtype="double")

    def detectFace(self, img):

        # faceBoxs
        facebox = []
        dets = self.detector(img, 1)
        print('dets: ', len(dets))
        if len(dets) == 1:
            for i, d in enumerate(dets):
                print("detected face:  {}".format(i + 1))
                print("faceBox: leftTop:({}, {}), rightBottom:({}, {})".format(d.left(), d.top(), d.right(), d.bottom()))
                facebox.append(d.left())
                facebox.append(d.top())
                facebox.append(d.right())
                facebox.append(d.bottom())
                cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (255, 0, 0), 2)
        return dets

    def landsFace68(self, dets, img):
        #  face Landmarks

        '''

        # 68关键点下标：
        head pose:
        鼻尖点：30
        右眼角外：36
        左眼角外：45

        右嘴角：48
        左嘴角：60
        下巴：8

        # 计算眼球中心：
        右眼角外：36
        右眼角内：39
        左眼角内：42
        左眼角外：45

        '''
        # 关键点下标
        index = [30, 36, 45, 48, 60, 8]
        eyelands = []
        headlands = []
        modellands = []
        predictor = dlib.shape_predictor(self.predictor_path)
        for k, d in enumerate(dets):
            shape = predictor(img, d)
            landmark = np.matrix([[p.x, p.y] for p in shape.parts()])
            print("face_landmark: {}".format(landmark.shape))
            # print(landmark)
            # head pose lands
            for i in index:
                headlands.append(landmark[i])
                modellands.append(self.model_points_68[i])
            # eye 眼球中心 lands
            reye = [int((landmark[36][0, 0] + landmark[39][0, 0]) / 2),
                    int((landmark[36][0, 1] + landmark[39][0, 1]) / 2)]
            leye = [int((landmark[42][0, 0] + landmark[45][0, 0]) / 2),
                    int((landmark[42][0, 1] + landmark[45][0, 1]) / 2)]
            eyelands.append(reye)
            eyelands.append(leye)

            for idx, point in enumerate(landmark):
                pos = (point[0, 0], point[0, 1])
                cv2.putText(img, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.20,
                            color=(0, 255, 0))
        # cv2.imwrite('1_color_landmarks_68.jpg', self.img)
        return eyelands, headlands, modellands

    def FaceEyeCrop(self, facebox, eyelands):
        # crop face
        facecrop = self.img[facebox[1]:facebox[3], facebox[0]:facebox[2]]  # 先y后x
        cv2.imwrite('1_color_facecrop.jpg', facecrop)
        # crop eye left-right
        rightEye = self.img[eyelands[0][1]-15:eyelands[0][1]+15, eyelands[0][0]-30:eyelands[0][0] + 30]
        leftEye = self.img[eyelands[1][1]-15:eyelands[1][1]+15, eyelands[1][0]-30:eyelands[1][0] + 30]
        cv2.imwrite('1_color_righteye.jpg', rightEye)
        cv2.imwrite('1_color_lefteye.jpg', leftEye)

    def get_obj_pts(self, img_pts, depth, camera_matrix):
        obj_pts = np.zeros(shape=(img_pts.shape[0], 3))
        for i, pt in enumerate(img_pts):
            x = int(pt[0])
            y = int(pt[1])
            d = depth[y][x]
            if d == 0:
                continue
            Z = d / 1000.
            Y = (y - camera_matrix[1][2]) * Z / camera_matrix[1][1]
            X = (x - camera_matrix[0][2]) * Z / camera_matrix[0][0]
            obj_pts[i][0] = X
            obj_pts[i][1] = Y
            obj_pts[i][2] = Z
        return obj_pts

    def get_euler_angle(self, rotation, translate):
        rotation_mat, _ = cv2.Rodrigues(rotation)
        pose_mat = cv2.hconcat((rotation_mat, translate))  # RT拼接4*3矩阵
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)  # 角度
        return pose_mat, euler_angle
        # return euler_angle

    def headPoseEst(self, dets, save_path, img):
        '''
        按顺序存储：
         head pose:
        鼻尖点：30     ---nose定为head世界零点
        右眼角外：36
        左眼角外：45

        右嘴角：48
        左嘴角：60
        下巴：8
        '''

        _, headlands, modellands = self.landsFace68(dets, img)
        # print("modellands: ", modellands)

        img_pts = np.array(headlands, dtype=np.float32)  # 2D点

        obj_pts = np.array(modellands, dtype=np.float32)  # 3D点

        # 坐标轴方向问题  ---待确认
        # for pts in obj_pts:
        #     pts[0] = -1*pts[0]
        #     pts[2] = -1*pts[2]

        # mm单位

        # 粗估的结果,误差太大
        # obj_pts = np.float32([(0.0, 0.0, 0.0),  # Nose tip
        #                       (-225.0, -170.0, 135.0),  # Right eye right corner
        #                       (225.0, -170.0, 135.0),  # Left eye left corner
        #                       (-150.0, 150.0, 125.0),  # Right mouth corner
        #                       (150.0, 150.0, 125.0),  # Left Mouth corner
        #                       (0.0, 330.0, 65.0)  # Chin
        #                       ])

        # head pose r,t
        _, rotation_vec, translation_vec = cv2.solvePnP(obj_pts, img_pts, self.cam_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        # print("rotation: ", rotation_vec)
        # print("translation: ", translation_vec)

        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose
        # x
        (nose_end_point2D_X, jacobian) = cv2.projectPoints(np.array([(100.0, 0.0, 0.0)]), rotation_vec,
                                                           translation_vec, self.cam_matrix, self.dist_coeffs)
        # y
        (nose_end_point2D_Y, jacobian) = cv2.projectPoints(np.array([(0.0, 100.0, 0.0)]), rotation_vec,
                                                           translation_vec, self.cam_matrix, self.dist_coeffs)
        # z
        (nose_end_point2D_Z, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 350.0)]), rotation_vec,
                                                           translation_vec, self.cam_matrix, self.dist_coeffs)

        # x轴：p1鼻尖点，p2反投影的点
        p1 = (int(img_pts[0][0][0]), int(img_pts[0][0][1]))
        p2 = (int(nose_end_point2D_X[0][0][0]), int(nose_end_point2D_X[0][0][1]))
        cv2.arrowedLine(img, p1, p2, (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.1)

        # y轴：
        p1 = (int(img_pts[0][0][0]), int(img_pts[0][0][1]))
        p2 = (int(nose_end_point2D_Y[0][0][0]), int(nose_end_point2D_Y[0][0][1]))
        cv2.arrowedLine(img, p1, p2, (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.1)

        # z轴：
        p1 = (int(img_pts[0][0][0]), int(img_pts[0][0][1]))
        p2 = (int(nose_end_point2D_Z[0][0][0]), int(nose_end_point2D_Z[0][0][1]))
        cv2.arrowedLine(img, p1, p2, (255, 0, 0), 2, cv2.LINE_AA, tipLength=0.1)

        # z
        # (nose_end_point2D_Z, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, -350.0)]), rotation_vec,
        #                                                    translation_vec, self.cam_matrix, self.dist_coeffs)
        # p1 = (int(img_pts[0][0][0]), int(img_pts[0][0][1]))
        # p2 = (int(nose_end_point2D_Z[0][0][0]), int(nose_end_point2D_Z[0][0][1]))
        # cv2.arrowedLine(img, p1, p2, (255, 255, 0), 2, cv2.LINE_AA, tipLength=0.1)



        # cv2.line(self.img, p1, p2, (255, 0, 0), 2)

        cv2.imwrite(save_path, img)
        return rotation_vec, translation_vec

    def headPose_EST(self,):
        fopen = open(self.data_path)  # ori_list
        lines = fopen.readlines()
        for line in lines:
            line = line.replace('\n', '')
            print(line)
            img_color = cv2.imread(line)
            line = line.split('/')
            sub_path = '/'.join(line[:-2])
            save_path = sub_path + '/' + line[-2] + '_pose'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            pose_name = line[-1][:-4] + '_pose.png'
            save_pose_path = save_path + '/' + pose_name

            dets = self.detectFace(img_color)
            if len(dets) == 0:  # 未检测到人脸
                continue
            else:
                r, t = self.headPoseEst(dets, save_pose_path, img_color)
                pose_rt, angles = self.get_euler_angle(r, t)

                # print("angles: ", angles)
                # print("pose_rt: ", pose_rt)

                # 验证头部坐标系的z轴朝向,取头部世界点(0,0,500)和(0,0,-500),使用post RT转到相机坐标系下,对比两个z值大小
                # z_500 = pose_rt[2][2] * 500 + pose_rt[2][3]
                # z_500_ = pose_rt[2][2] * (-500) + pose_rt[2][3]
                # print("z 500:", z_500)
                # print("z -500:", z_500_)

def main():
    #
    facedetector = faceLandMarksDect('./shape_predictor_68_face_landmarks.dat', './camera_param.txt', 'test.txt')
    facedetector.headPose_EST()
    # 弧度制转为角度
    # angles = self.get_euler_angle(r, t)  # 此时得到角度值
    # print("angles: ", angles)
    # pitch, yaw, roll = [math.radians(_) for _ in angles]  # 此时将角度转为弧度制
    # pitch = math.degrees(math.asin(math.sin(pitch)))
    # yaw = math.degrees(math.asin(math.sin(yaw)))
    # roll = -math.degrees(math.asin(math.sin(roll)))
    # print("pitch: ", pitch)
    # print("yaw: ", yaw)
    # print("roll: ", roll)



if __name__ == "__main__":
    main()
