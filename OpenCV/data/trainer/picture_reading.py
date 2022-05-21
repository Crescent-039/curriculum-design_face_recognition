
import cv2 as cv
import win32api

def face_detect_demo(resize_img):      #检测函数
    gray = cv.cvtColor(resize_img, cv.COLOR_BGR2GRAY)      #转灰度图像
    #调用检测函数
    face_detect = cv.CascadeClassifier('D:/opencv/opencv/build/etc/haarcascades/haarcascade_frontalface_alt2.xml')
    face = face_detect.detectMultiScale(gray)
    for x, y, w, h in face:
        cv.rectangle(resize_img, (x,y), (x+w, y+h), color=(0, 0, 255), thickness=2)
    cv.imshow('result', resize_img)


#cap = cv.VideoCapture(0, cv.CAP_DSHOW)                #读取摄像头

img = cv.imread('face01.jpg')  # 读图
#resize_img = cv.resize(img, dsize=(600, 600))  # 修改图片尺寸

#face_detect_demo() #检测函数

cv.imshow('img', img)  # 显示原图
#cv.imshow('resize_img', resize_img)  # 显示修改后的
#print('未修改', img.shape)  # 打印原图
#print('已修改', resize_img.shape)  # 打印修改尺寸的图

# gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  #灰度转换
# cv.imshow('gray', gray_img)                     #显示灰度
# cv.imwrite('gray_face01.jpg', gray_img)         #保存灰度图片
# cv.imshow('read_img', img)                      #显示图
while True:

    esc = win32api.GetKeyState(27)                #键盘识别 按Esc跳出
    if esc < 0:
        print('esc')        #证明已经识别按键
        break
    else:
        cv.waitKey(0)
cv.destroyAllWindows()  # 释放内存

