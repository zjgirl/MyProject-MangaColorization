import numpy as np
import cv2

class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y

def getGrayDiff(img, currentPoint, tmpPoint):
    return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))


def selectConnects(p):
    if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
                    Point(0, 1), Point(-1, 1), Point(-1, 0)]
    else:
        connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
    return connects

def regionGrow(img, seeds, thresh, p=1):
    height, weight = img.shape
    seedMark = np.zeros(img.shape)
    seedList = []
    for seed in seeds:
        seedList.append(seed)
    label = 1
    connects = selectConnects(p)
    while (len(seedList) > 0):
        currentPoint = seedList.pop(0)

        seedMark[currentPoint.x, currentPoint.y] = label
        for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
            if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
                seedMark[tmpX, tmpY] = label
                seedList.append(Point(tmpX, tmpY))
    return seedMark

def on_mouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Start Mouse Position: ' + str(x) + ', ' + str(y))
        seeds.append(Point(y, x))


sketch = cv2.imread('5_s.jpg', 0)
color = cv2.imread('5_c.jpg')

sketch = cv2.resize(sketch,(512,512))
_,sketch = cv2.threshold(sketch, 245, 255, cv2.THRESH_BINARY)
seeds = []

cv2.namedWindow('input')
cv2.setMouseCallback('input', on_mouse, 0)
cv2.imshow('input', sketch)
cv2.waitKey()

binaryImg = regionGrow(sketch, seeds, 1)
color[np.where(binaryImg == 1)] = (255,255,255)
cv2.imshow(' ', binaryImg)
cv2.waitKey(0)
cv2.imshow(' ', color)
cv2.waitKey(0)
cv2.imwrite("seg.jpg", color)