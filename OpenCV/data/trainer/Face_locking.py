import cv2 as cv
import win32api


def face_detect_demo(Vshow):  # 检测函数
    gray = cv.cvtColor(Vshow, cv.COLOR_BGR2GRAY)  # 转灰度图像
    # 调用检测函数
    face_detect = cv.CascadeClassifier('D:/opencv/opencv/build/etc/haarcascades/haarcascade_frontalface_alt2.xml')
    face = face_detect.detectMultiScale(gray)
    for x, y, w, h in face:
        cv.rectangle(Vshow, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
    cv.imshow('result', Vshow)


cap = cv.VideoCapture(0, cv.CAP_DSHOW)  # 读取摄像头

flag = 1
num = 1

while cap.isOpened():  # 检测是否开启
    ret_flag, Vshow = cap.read()  # 得到每帧图像
    cv.imshow("Capture_Test", Vshow)  # 显示每帧图像
    face_detect_demo(Vshow)
    k = cv.waitKey(1) & 0xFF  # 按键判断
    if k == ord('s'):  # 保存
        cv.imwrite("D:/pycharm/Face recognition/OpenCV/data/picture/" + str(num) + ".name" + ".jpg", Vshow)
        print("success to save" + str(num) + ".jpg")
        print("-----------")
        num += 1
    elif k == int('27'):
        break

cap.release()
cv.destroyAllWindows()
