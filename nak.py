import numpy as np
import cv2

def check_inout(img, data_list, point_list):
    center_point_x = img.shape[0]/2

    inout_list = []

    for i in point_list:

        for j in data_list:
            if j[0]<center_point_x:
                xy = [j[2],j[3]]
            else:
                xy = [j[0],j[3]]
            if (xy[0] > i[0]) and (xy[0] < i[2]) and (xy[1] < i[3]) and (xy[1] > i[1]):
                a = 1
                break
            else:
                a = 0
        inout_list.append(a)
        
    print(inout_list)
    return inout_list

data_list = [[197, 65, 268, 117], [302, 23, 324, 63], [623, 74, 719, 144], [595, 68, 641, 113], [56, 76, 252, 165], [453, 66, 607, 128], [0, 175, 155, 354], [0, 133, 190, 234]]
point_list = [[216,111,289,145],[168,150,261,187],[122,201,220,248],[74,254,205,311],[25,322,183,418],[434,113,518,132],[449,136,530,151],[467,156,588,187],[497,194,647,233],[544,245,682,297],[582,316,719,389]]

img = cv2.imread("data/images/def/abs.png")

check_inout(img,data_list,point_list)

    



def check_parking(img, img2, xyxy, src, dst):
    center_point_x = img.shape[0]/2
    h1, status = cv2.findHomography(dst, src)
    a = 10  
    img2_copy = img2.copy()
    img2 = cv2.resize(img2_copy, dsize=(0,0),fx=1/a,fy=1/a, interpolation=cv2.INTER_LINEAR)        

    for i in xyxy:
        x = i
        if x[0] < center_point_x:
            xy = [x[2],x[3],1]
            point = np.dot(h1,xy)
            parking_point = point/point[2]
        else:
            xy = [x[0],x[3],1]
            point = np.dot(h1,xy)
            parking_point = point/point[2]
    
        parking_point = np.delete(parking_point, 2)
        parking_point = np.array(parking_point/a, dtype=int)
        cv2.circle(img2, parking_point,2, (0,255,0),-1)
        print(parking_point * a)

    cv2.imshow("detecting_point",img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#매칭점
#1
# pts_src = np.array([[7736, 1688], [7396, 1686], [7402, 1439], [7720, 1442]])
# pts_dst = np.array([[202, 182], [259, 117], [448, 118], [506, 183]])
# #2
# pts_src = np.array([[7443,1699],[7163,1427],[7945,1425],[7943,1699]])
# pts_dst = np.array([[263,122],[424,91],[650,405],[54,406]])

# h1, status = cv2.findHomography(pts_dst, pts_src)
# print("homography : ", h1)
# print("Homography status : ", status)

# data_list = [[197, 65, 268, 117], [302, 23, 324, 63], [623, 74, 719, 144], [595, 68, 641, 113], [56, 76, 252, 165], [453, 66, 607, 128], [0, 175, 155, 354], [0, 133, 190, 234]]
# img = cv2.imread("80_line.png")
# img2 = cv2.imread("ch_b1_point.png")
# center_point_x = img.shape[0]/2

# check_parking(img,img2,data_list,pts_src,pts_dst)
