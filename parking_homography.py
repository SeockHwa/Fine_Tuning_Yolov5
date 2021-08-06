import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage import transform

# src = [[66, 425], [291, 119], [464, 125], [676, 436]]
# dst = [[200,200],[200,1800],[2300 , 1800],[2300, 200]]
# src = np.array(src)
# dst = np.array(dst)
# homo_matrix, tmp = cv2.findHomography(dst, src)

# print(homo_matrix)

palawan = imread('data/images/input_image/1/3.png')
imshow(palawan)
def project_planes(image, src, dst):
    x_src = [val[0] for val in src] + [src[0][0]]
    y_src = [val[1] for val in src] + [src[0][1]]
    x_dst = [val[0] for val in dst] + [dst[0][0]]
    y_dst = [val[1] for val in dst] + [dst[0][1]]
    
    fig, ax = plt.subplots(1,2, figsize=(13,6))
    
    new_image = image.copy() 
    projection = np.zeros_like(new_image)
    ax[0].imshow(new_image)
    ax[0].plot(x_src, y_src, 'r--')
    ax[0].set_title('Area of Interest')
    ax[1].imshow(projection)
    ax[1].plot(x_dst, y_dst, 'r--')
    ax[1].set_title('Area of Projection')
    

area_of_interest = [(66, 425),
                    (291, 119),
                    (464, 125),
                    (676, 436)]
area_of_projection = [(200, 200),
                      (200, 1800),
                      (2300, 1800),
                      (2300, 200)]
project_planes(palawan, area_of_interest, area_of_projection)