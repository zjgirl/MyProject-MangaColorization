import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

#本程序描述坐标均是(y, x),y表示纵坐标，x表示横坐标

img_path = "./imgs/sample_7.jpg"
eye_path = "./imgs/eyes3.jpg"

def detectFace(filename, cascade_file="lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):  # lbpcascade_animeface.xml文件可在github上面找到，就是一个巨长的xml格式代码，表示看不懂。
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(48, 48))
    res = []
    for i, (x, y, w, h) in enumerate(faces):
        if w > 100:  # 过滤掉一些错检的人脸
            res.append((x, y, w, h))
            face = image[y: y + h, x:x + w, :]
            cv2.rectangle(image,(x, y), (x+w, y+h), (255, 255, 0), 5)
            cv2.imshow("AnimeFaceDetect", image)
            cv2.waitKey(0)
    return res  # 返回人脸, 左上角坐标+长宽


def locateEyes(img_path, face):
    eyes = []
    image = cv2.imread(img_path)
    while (face):
        x, y, w, h = face.pop()
        img = image[y: y + h, x:x + w, :]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 根据face的大小，统计发现大概<200, minRadius=10, maxRadius = 30
        # > 200(maybe < 300 or 400), minRadius=20, maxRadius = 40
        radius = []
        if (w < 200):
            radius = [10, 30]
        elif w < 400:
            radius = [20, 40]

        #根据曲率找出可能的圆，返回其圆心
        circles1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,
                                    60, param1=100, param2=30, minRadius=radius[0], maxRadius=radius[1])
        circles = circles1[0, :, :] #(x, y, radius)
        circles = np.uint16(np.around(circles))
        for i in circles[:]:
            #cv2.circle(image, (i[0]+x, i[1]+y), i[2], (0, 0, 255), 2) #画外圆
            #cv2.circle(image, (i[0]+x, i[1]+y), 2, (255, 0, 255), 2) #画中心
            #cv2.rectangle(img, (i[0] - i[2], i[1] + i[2]), (i[0] + i[2], i[1] - i[2]), (255, 255, 0), 5)
            print("Eyes Position", i[0]+x, i[1]+y)
            eyes.append([i[1] + y, i[0] + x])  # 要找出其在原图上的位置

        #cv2.imshow("LocateEyes", image)
        #cv2.waitKey(0)

    return eyes

seed = locateEyes(img_path, detectFace(img_path))
def on_mouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print ('Eyes Position: ' + str(y) + ', ' + str(x))
        seed.append([y, x])

def find_region(img, img_rg, start, eyes):
    # print 'starting points',i,j
    x = [-1, 0, 1, -1, 1, -1, 0, 1]  # 指示八邻域位置
    y = [-1, -1, -1, 0, 0, 1, 1, 1]

    rows, columns = np.shape(img_rg)
    region_points = [start]

    maxX = 0; maxY = 0
    minX = 9999; minY = 9999

    while (len(region_points) > 0):  # 当种子列表不为空
        point = region_points.pop(0)  # 将种子列表中第0个像素取出作为种子，并移除这个像素
        i = point[0]  # y
        j = point[1]  # x
        # print 'value of pixel',val
        for k in range(8):  # 检查种子像素的8邻域
            # print '\ncomparison val:',val, 'ht',ht,'lt',lt
            if 0 <= i + y[k] < rows and 0 <= j + x[k] < columns:  # 该邻域像素在图像范围内
                if img[i + y[k]][j + x[k]] == eyes and img_rg[i + y[k]][j + x[k]] != 1:  # 区域生长的条件：邻域像素的灰度值在当前像素值的+-8之内
                    # print '\nbelongs to region',arr[i+x[k]][j+y[k]]
                    img_rg[i + y[k]][j + x[k]] = 1  # 邻域值被赋值为1
                    region_points.append([i + y[k], j + x[k]])  # 将邻域像素加入到种子列表中
                    # if [i+y[k],j+x[k]] not in region_points: #如果该邻域像素未被添加到种子列表中

                    #计算中心坐标
                    if i + y[k] < minY: minY = i + y[k]
                    if j + x[k] < minX: minX = j + x[k]
                    if i + y[k] > maxY: maxY = i + y[k]
                    if j + x[k] > maxX: maxX = j + x[k]
    center = [(minY + maxY) // 2, (minX + maxX) // 2]
    return img_rg, center

# 获得一个球形大小的结构元素，用作腐蚀膨胀的内核
def get_ball_structuring_element(radius):
    """Get a ball shape structuring element with specific radius for morphology operation.
    The radius of ball usually equals to (leaking_gap_size / 2).

    # Arguments
        radius: radius of ball shape.

    # Returns
        an array of ball structuring element.
    """
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    # 核为椭圆形：MORPH_ELLIPSE;参数二为内核的尺寸

def find_img_eyes(img_path, eyes, need_sketch):

    img = cv2.imread(img_path)
    h, w, _ = np.shape(img)

    cv2.namedWindow('img', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('img', on_mouse, 0)

    for i in seed:
        #cv2.circle(img, (i[1], i[0] ), i[2], (0, 0, 255), 2)  # 画外圆
        cv2.circle(img, (i[1], i[0] ), 2, (0, 0, 255), 2)  # 画中心
    cv2.imshow('img', img)
    cv2.waitKey(0)

    img_rg = np.zeros((h, w))
    for i in range(len(seed)):
        img_rg[seed[i][0]][seed[i][1]] = 1

    if need_sketch:
        sketch_path =  os.path.split(img_path)[0] + "/sketch_" + img_path.split('_')[1]
        img = cv2.imread(sketch_path)

    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    _,img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)

    center = []
    while seed:
        start = seed.pop()
        img_rg, center_point = find_region(img, img_rg, start, eyes) #找到眼睛区域以及眼睛的中心位置
        center.append(center_point)
        #cv2.circle(img_rg, center, 50, (255, 0, 0))

    #膨胀腐蚀
    ball = get_ball_structuring_element(2)
    img_rg = cv2.morphologyEx(img_rg, cv2.MORPH_DILATE, ball, anchor=(-1, -1), iterations=2)
    img_rg = cv2.morphologyEx(img_rg, cv2.MORPH_ERODE, ball, anchor=(-1, -1), iterations=2)

    cv2.imshow('mask',img_rg)
    cv2.waitKey(0)

    return img_rg, center

#输入参数分别为，彩色图，眼睛图，彩色图眼睛位置图，眼睛中心点位置，眼睛图眼睛位置图，眼睛中心点位置
#copy的方法采用区域生长，由中心位置开始生长
def copy_eyes(img, eye, img_eyes, img_eyes_center, choose_eye, choose_eye_center):
    x = [-1, 0, 1, -1, 1, -1, 0, 1]  # 指示八邻域位置
    y = [-1, -1, -1, 0, 0, 1, 1, 1]
    rows, columns, _ = np.shape(img)
    while img_eyes_center:
        img_eye_center = img_eyes_center.pop()
        img_points = [img_eye_center]
        eye_points = [[choose_eye_center[0][0], choose_eye_center[0][1]]]
        img_rg = np.zeros((rows, columns))
        while (len(img_points) > 0):  # 当种子列表不为空
            img_point = img_points.pop(0)  # 将种子列表中第0个像素取出作为种子，并移除这个像素
            eye_point = eye_points.pop(0)
            img_i = img_point[0]  # y
            img_j = img_point[1]  # x
            eye_i = eye_point[0]
            eye_j = eye_point[1]
            for k in range(8):  # 检查种子像素的8邻域
                if img_eyes[img_i + y[k]][img_j + x[k]] == 1 and choose_eye[eye_i + y[k]][eye_j + x[k]] == 1 and img_rg[img_i + y[k]][img_j + x[k]] != 1:
                    img_rg[img_i + y[k]][img_j + x[k]] = 1  # img_rg用于指示该像素是否被赋值过
                    img[img_i + y[k]][img_j + x[k]] = eye[eye_i + y[k]][eye_j + x[k]] #眼睛copy
                    img_points.append([img_i + y[k], img_j + x[k]])  # 将邻域像素加入到种子列表中
                    eye_points.append([eye_i + y[k], eye_j + x[k]])
    return img


if __name__ == "__main__":

    img_eyes, img_eyes_center = find_img_eyes(img_path, 255, True) #图像中白色区域是眼睛
    choose_eye, choose_eye_center = find_img_eyes(eye_path, 0, False) #眼睛图中黑色区域是眼睛

    img = cv2.imread(img_path)
    eye = cv2.imread(eye_path)
    img = copy_eyes(img, eye, img_eyes, img_eyes_center, choose_eye, choose_eye_center)

    cv2.imshow('output', img)
    cv2.waitKey(0)
    cv2.imwrite(os.path.split(img_path)[0] + "/output_" + img_path.split('_')[1], img)

