import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('../imagens/amostras/zoom/skeletonized/skeletonized7.jpg',0)             # queryImage
img2 = cv2.imread('../imagens/amostras/zoom/skeletonized/skeletonized8.jpg', 0)             # trainImage

# img1 = cv2.resize(img1,(400,400),interpolation=cv2.INTER_AREA)
# img2 = cv2.resize(img2,(400,400),interpolation=cv2.INTER_AREA)

# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:40],outImg=None, flags=2)

cv2.imshow("desenho", img3)
print("jj"+ str(img3))
plt.imshow(img3),plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()