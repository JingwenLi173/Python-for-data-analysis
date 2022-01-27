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

# subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)


# blur and threshold the image
blurred = cv2.blur(gradient, (9, 9))
(_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
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

(cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
print('c', c.shape)

# compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)
box = np.int0(cv2.boxPoints(rect)) # cv2.cv.BoxPoints was changed. For OpenCV 3.x, use cv2.boxPoints instead.

# draw a bounding box arounded the detected barcode and display the image
cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
# cv2.imshow("Image", image)
cv2.imwrite("contoursImage2.jpg", image)
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
