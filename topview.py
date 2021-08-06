import cv2
from operator import itemgetter
from glob import glob
import matplotlib.pyplot as plt
import numpy as np

input = cv2.imread('data/images/input_image/1/input_image.png')
cv2.imshow("input",input)
# Coordinates that you want to Perspective Transform
pts1 = np.float32([[1, 719], [537, 135], [898, 135], [1464, 719]])
# Size of the Transformed Image
pts2 = np.float32([[100,380],[100,100],[620,100],[620,380]])
#for val in pts1:
#    cv2.circle(paper,(val[0],val[1]),5,(0.0,255.0,0.0),-1)
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(input,M,(720,480))
dst = np.array(dst)
cv2.imshow("dst",dst)
cv2.waitKey()