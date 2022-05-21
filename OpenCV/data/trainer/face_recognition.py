import os
import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()  # 加载识别器
recognizer.read('D:/pycharm/Face recognition/OpenCV/data/trainer/FaceDataTrainer20191726.yml')  # 加载训练好的数据

names = []  # 名称
warningtime = 0  # 警告全局变量
sec = 0        #识别时间变量
rnum = 0       #是否识别到人类的变量


def md5(str):  # md加密模块
    import hashlib
    m = hashlib.md5()
    m.update(str.encode("utf8"))
    return m.hexdigest()


def face_detect_demo(img):  # 检测函数
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转灰度图像
    # 调用检测函数
    face_detect = cv2.CascadeClassifier('D:/opencv/opencv/build/etc/haarcascades/haarcascade_frontalface_alt2.xml')
    face = face_detect.detectMultiScale(gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (100, 100), (300, 300))
    for x, y, w, h in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
        cv2.circle(img, center=(x + w // 2, y + h // 2), radius=w // 2, color=(255, 0, 0), thickness=2)
        global ids
        ids, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        print('id:', ids, '评分：', confidence)
        global rnum

        rnum += 1

        if confidence < 30:  # confidence值越大，越不可信？我感觉是越小越不可信
            global warningtime
            warningtime += 1
            if warningtime > 100:
                warningtime = 0
            cv2.putText(img, 'Warning!This student is not ',
                        (x + 10, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
            cv2.putText(img, 'belong to this classroom!',
                        (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
        else:
            cv2.putText(img, str(names[ids - 1]), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
    cv2.imshow('result', img)


def name():         #遍历图片中的名字
    path = 'D:/pycharm/Face recognition/OpenCV/data/pictures_for_training/'
    # names = []
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        name = str(os.path.split(imagePath)[1].split('.', 2)[1])
        names.append(name)


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 读取摄像头
name()
while True:
    flag, frame = cap.read()
    if not flag:  # 如果没有读到摄像头则跳出
        break
    face_detect_demo(frame)
    if rnum == 0:  # 根据rnum值来判断是否识别到人
        print('没有识别到人类，请再次识别')
        break
    if sec < 90:  # 识别三秒后跳出
        sec += 1
    else:
        print('这个学生是:', ids, names[ids - 1])
        break
    k = cv2.waitKey(1) & 0xFF  # 按键判断
    if k == int('27'):              #按下ESC后跳出
        break
cv2.destroyAllWindows()
cap.release()
# print(names)
