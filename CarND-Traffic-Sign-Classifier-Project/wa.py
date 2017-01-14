import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import numpy as np
import matplotlib.image as mpimg

def normalize(image):
    return (image - image.mean()) / (image.std() + 1e-8)

def generateImg(img):
    
    rot_range = np.random.randint(10, high=50)
    shear_range = np.random.randint(1, high=10)
    trans_range = np.random.randint(1, high=5)
    gray = np.random.randint(0, high=100)%2
    
    rot_angle = np.random.normal(rot_range)-rot_range/2
    rows,cols,ch = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),rot_angle,1)
    img = cv2.warpAffine(img,Rot_M,(cols,rows))

    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    img = cv2.warpAffine(img,Trans_M,(cols,rows))

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])
    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2
    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])
    shear_M = cv2.getAffineTransform(pts1,pts2)
    img = cv2.warpAffine(img,shear_M,(cols,rows))

    if gray == 1:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return img

image = cv2.imread('stopsign.jpg')
img1 = normalize(image)

cv2.imshow('img', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()