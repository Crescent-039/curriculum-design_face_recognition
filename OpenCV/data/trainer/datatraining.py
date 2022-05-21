import os
import cv2
from PIL import Image
import numpy as np


def getImageAndLabels(path):
    global id
    facesSamples = []  # 储存人脸数据
    ids = []  # 储存姓名
    ImagePaths = [os.path.join(path, f) for f in os.listdir(path)]  # 储存图片信息
    face_detect = cv2.CascadeClassifier('D:/opencv/opencv/build/etc/haarcascades/haarcascade_frontalface_alt2.xml')
    for ImagePath in ImagePaths:  # for in循环，用ImagePath遍历ImagePaths中的每一个元素
        PIL_img = Image.open(ImagePath).convert('L')  # 打开图像并将其灰度化
        img_numpy = np.array(PIL_img, 'uint8')  # 将黑白图像转为数组
        faces = face_detect.detectMultiScale(img_numpy)  # 获取人脸特征
        id = int(os.path.split(ImagePath)[1].split('.')[0])  # 获取每张图片的id和姓名
        for x, y, w, h in faces:
            ids.append(id)
            facesSamples.append(img_numpy[y:y + h, x:x + w])  # 预防无脸图片

    print('id:', id)
    print('fs:', facesSamples)
    return facesSamples, ids


if __name__ == '__main__':
    path = 'D:/pycharm/Face recognition/OpenCV/data/pictures_for_training/'  # 图片路径
    faces, ids = getImageAndLabels(path)  # 获取图像数组和id数组和签名
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # 加载识别器
    recognizer.train(faces, np.array(ids))  # 训练
    recognizer.write('D:/pycharm/Face recognition/OpenCV/data/trainer/FaceDataTrainer20191726.yml')  # 保存
