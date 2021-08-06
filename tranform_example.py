# -*- coding: utf-8 -*-
import numpy as np
import cv2
import argparse

# true if mouse is pressed
drawing = False 
#rot90 = False

src_x, src_y = -1,-1
dst_x, dst_y = -1,-1

src_list = []
dst_list = []

parser = argparse.ArgumentParser()
parser.add_argument("--rot", default = False, help='show the stitched image')
parser.add_argument("--imgPath", '-i', type=str, help='image path')

def select_points_src(event,x,y,flags,param):
    global src_x, src_y, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        src_x, src_y = x,y
        cv2.circle(src_copy,(x,y),5,(0,0,255),-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def select_points_dst(event,x,y,flags,param):
    global dst_x, dst_y, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        dst_x, dst_y = x,y
        cv2.circle(dst_copy,(x,y),5,(0,0,255),-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def get_plan_view(src, dst):
    global rot90
    src_pts = np.array(src_list).reshape(-1,1,2)
    dst_pts = np.array(dst_list).reshape(-1,1,2)
    h,w = src.shape[:2]
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    print("Homography:")
    print(H)
    if rot90 == True:
        src = cv2.rotate(src, cv2.ROTATE_90_COUNTERCLOCKWISE)  
        plan_view = cv2.warpPerspective(src, H, (h*2,w*2))
    else:
        plan_view = cv2.warpPerspective(src, H, (w*2,h*2))
        print(src.shape)
    return plan_view

	

FLAGS = parser.parse_args()
rot90 = FLAGS.rot



########################################################################
num = 71
imgpath = 'data/images/input_image/1/input_image.png'
# [241, 890], [241, 890], [723, 358], [1646, 333]
########################################################################



########변환 이미지
#src = cv2.imread('imgs/data_2/4.png', -1)
src = cv2.imread(imgpath, -1)
#src = cv2.resize(src, dsize=(0,0),fx=0.8,fy=0.8, interpolation=cv2.INTER_LINEAR)
src_copy = src.copy()
cv2.namedWindow('src')
cv2.moveWindow("src", 80,80)
cv2.setMouseCallback('src', select_points_src)

########기준 이미지
dst = cv2.imread('white.png', -1)
#dst = cv2.resize(dst, dsize=(0,0),fx=0.8,fy=0.8, interpolation=cv2.INTER_LINEAR)
dst_copy = dst.copy()
cv2.namedWindow('dst')
#cv2.moveWindow("dst", 780,80);
cv2.setMouseCallback('dst', select_points_dst)

saveCnt = 0

while(1):
    cv2.imshow('src',src_copy)
    cv2.imshow('dst',dst_copy)
    k = cv2.waitKey(1) & 0xFF
    ######이미지마다 4개의 포인트 지정, 매 포인트 위치 클릭후 세이브 진행
    ######변환 이미지 부터 포인트 저장
    if k == ord('s'):
        print('save points')
        saveCnt += 1


        if saveCnt > 4:
            cv2.circle(dst_copy,(dst_x,dst_y),5,(0,255,0),-1)
            dst_list.append([dst_x,dst_y])
            print("dst points:")
            print(dst_list)
        else:
            cv2.circle(src_copy,(src_x,src_y),5,(0,255,0),-1)
            src_list.append([src_x,src_y])

            print("src points:")
            print(src_list)
    elif k == ord('h'):
        print('create plan view')
        print(rot90)
        plan_view = get_plan_view(src, dst)
        if plan_view.shape[2]>3:
            b = plan_view[:,:,0]
            g = plan_view[:,:,1]
            r = plan_view[:,:,2]
            plan_view = cv2.merge((b,g,r))
        cv2.imshow("plan view", plan_view) 
        cv2.imwrite("result_sink_%d.png"%num, plan_view)
    elif k == 27:
        break

cv2.destroyAllWindows()