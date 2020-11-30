import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('amostras/20201104_131527 (2).jpg',0)

# Initiate STAR detector
orb = cv2.ORB_create()

# find the keypoints with ORB
kp = orb.detect(img,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img,kp,img ,color=(0,255,0), flags=0)
plt.imshow(img2),plt.show()