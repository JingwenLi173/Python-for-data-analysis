import cv2
import numpy as np
 
 
def ROI_byMouse(img,lsPointsChoose):
    mask = np.zeros(img.shape, np.uint8)
    pts = np.array(lsPointsChoose, np.int32)  # pts是多边形的顶点列表（顶点集）
    col0 =pts[:,0]
    col1 =pts[:,1]
    x1=np.min(col0)
    y1=np.min(col1)
    x2=np.max(col0)
    y2 = np.max(col1)
    pts = pts.reshape((-1, 1, 2))
    # 这里 reshape 的第一个参数为-1, 表明这一维的长度是根据后面的维度的计算出来的。
    # OpenCV中需要先将多边形的顶点坐标变成顶点数×1×2维的矩阵，再来绘制
 
    # --------------画多边形---------------------
    mask = cv2.polylines(mask, [pts], True, (255, 255, 255))
    ##-------------填充多边形---------------------
    mask2 = cv2.fillPoly(mask, [pts], (255, 255, 255))
    ROI = cv2.bitwise_and(mask2, img)
 
    return  ROI[y1:y2,x1:x2]
 
 
 

points =  [
    [
        189.1232876712329,
        117.12328767123287
    ],
    [
        177.47945205479454,
        115.06849315068493
    ]...
]
img = cv2.imread('1.jpg')

img = ROI_byMouse(img,points)
cv2.imshow('cut_out',img)
cv2.waitKey(0)
cv2.destroyWindow('cut_out')
