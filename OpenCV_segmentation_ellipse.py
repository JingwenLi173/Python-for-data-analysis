#!/usr/bin/env python
# -*- coding: utf-8 -*-

########## Jingwen Li 01/27/2022 ##########
#check your working directory
#import os
#os.getcwd()
#os.chdir('/Users/jingwenli/Downloads/Xiaokang_data')
#os.getcwd()
#os.listdir()
########## ########## ########## ##########
# Untargeted metabolomics
# samples: 231 pos 644 neg 5 none
########## ########## ########## ##########

import cv2
import pandas as pd
import numpy as np
import os
print(os.getcwd())

########## ########## ########## ##########

# 1. transform to gray
image = cv2.imread("./img_dir/R1T1.jpg")
print(image.shape)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(gray.shape)
# cv2.imshow('gray', gray)
# cv2.destroyAllWindows()
cv2.imwrite("gray.jpg", gray)

gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

# 3. subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)


# 3. blur and threshold the image
blurred = cv2.blur(gradient, (9, 9))
(_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY) # get binary/black-white image
print('blurred', blurred.shape)
# cv2.imshow('blurred', blurred)
cv2.imwrite("blurred.jpg", blurred)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
print('closed1', closed.shape)
# cv2.imshow('closed1', closed)
cv2.imwrite("closed1.jpg", closed)

# perform a series of erosions and dilations
closed = cv2.erode(closed, None, iterations=4)
closed = cv2.dilate(closed, None, iterations=4)
print('closed2', closed.shape)
# cv2.imshow('closed2', closed)
cv2.imwrite("closed2.jpg", closed)

# find the contours: cv2.findContours() must be binary/black-white image
(cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(cnts)) # how many contours
# print(cnts[0])
# print('c', c.shape)
c = sorted(cnts, key=cv2.contourArea, reverse=True)
for i in range(len(cnts)):
    area = cv2.contourArea(c[i])
    # print('area',i, area)
    if area > 17000:
        hull = cv2.convexHull(c[i])
        # print(hull) #轮廓的索引点
        image = cv2.polylines(image, [hull], True, (255, 0, 0), 8)
        
        cv2.waitKey(0) 
    else:
        pass
    cv2.imwrite("image_ellipse.jpg", image)
    
'''
#1. cv2.findContours()函数第一个参数是要检索的图片，必须是为二值图，即黑白的（不是灰度图），所以读取的图像要先转成灰度的，再转成二值图，我们在第二步用cv2.threshold()函数已经得到了二值图。第二个参数表示轮廓的检索模式。第三个参数为轮廓的近似方法。

#2. cv2.findContours()的第二个参数是轮廓的检索模式，有以下4种：
cv2.RETR_EXTERNAL表示只检测外轮廓
cv2.RETR_LIST检测的轮廓不建立等级关系
cv2.RETR_CCOMP建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
cv2.RETR_TREE建立一个等级树结构的轮廓。

#3. cv2.findContours()的第三个参数是轮廓的近似方法，即采用多少个点来描述一个轮廓：
cv2.CHAIN_APPROX_NONE存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1#
cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息

#4. cv2.findContours()返回的两个参数：一个是轮廓本身，还有一个是每条轮廓对应的属性，通常我们只使用第一个参数
(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
其中cnts是一个list，通过print(len(list))可以查看共检测出多少个轮廓，如下图轮廓框的个数=16，
print(cnts[0])打印出第一个轮廓相应的描述点
'''




'''
# (1) compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)
box = np.int0(cv2.boxPoints(rect)) # cv2.cv.BoxPoints was changed. For OpenCV 3.x, use cv2.boxPoints instead.
cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

# (2) draw arounded the detected barcode and display the image
cv2.drawContours(image, c, -1, (0, 255, 0), 3)
# cv2.imshow("Image", image)
cv2.imwrite("contoursImage2.jpg", image)
cv2.waitKey(0)

1. 利用rect0=cv2.minAreaRect()函数求得包含点集最小面积的矩形，这个矩形是可以有偏转角度的，可以与图像的边界不平行。
2. box = np.int0(cv2.boxPoints(rect))返回的box是对应矩阵的4个坐标
3. 这里我们利用box0，box1的坐标信息，求出包含这两个矩形的最合理的坐标
4. cv2.drawContours(all_box_img, cnts, -1, (0, 255, 0), 1)
   cv2.drawContours(top2_box_img, [box0,box1], -1, (0, 255, 0), 1)
第一个参数是指明在哪幅图像上绘制轮廓
第二个参数是是轮廓本身(.dtype=list)(如上面step3得到的cnts就是形状不规则的轮廓，step4得到的[box0,box1]就是最小外接矩阵）
第三个参数指定绘制轮廓list中的哪条轮廓，如果是-1，则绘制其中的所有轮廓，可以输出0，1，2等
第四个参数是轮廓线条的颜色
第五个参数是轮廓线条的粗细

# (3) 轮廓的近似
epsilon = 0.05*cv2.arcLength(c[0],True)
print('epsilon', epsilon)
approx = cv2.approxPolyDP(c[0],epsilon,True)
cv2.polylines(image, [approx], True, (0, 255, 0), 2)
cv2.imwrite("image_ellipse.jpg", image)
cv2.waitKey(0)

Xs = [i[0] for i in box]
Ys = [i[1] for i in box]
x1 = min(Xs)
x2 = max(Xs)
y1 = min(Ys)
y2 = max(Ys)
hight = y2 - y1
width = x2 - x1
cropImg = image[y1:y1+hight, x1:x1+width]
cv2.imwrite("cropImg.jpg", cropImg)
'''
