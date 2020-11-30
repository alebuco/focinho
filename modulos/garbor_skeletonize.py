import glob
import cv2
import os
import numpy as np

def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters

def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

def gravaImagem(imagem, caminho):
    cv2.imwrite(caminho, imagem)

    return 1

# img_fn = '../imagens/amostras/zoom/'
n = 0
x = 0
garbo = "../imagens/amostras/zoom/garbo/"
skeleton = "../imagens/amostras/zoom/skeletonized/"

for img_fn in glob.glob(os.path.join("../imagens/amostras/zoom/", "*.jpg")):
    imgOriginal = cv2.cvtColor(cv2.imread(img_fn),cv2.COLOR_BGR2GRAY)
    img = imgOriginal

    filters = build_filters()

    res1 = process(img, filters)
    res1 = cv2.bitwise_not(res1)
    img = res1

    # Threshold the image
    ret, img = cv2.threshold(img, 170, 255, 0)

    # cv2.waitKey(0)

    # Step 1: Create an empty skeleton
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    # Get a Cross Shaped Kernel
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # Repeat steps 2-4
    while True:
        # Step 2: Open the image
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        # Step 3: Substract open from the original image
        temp = cv2.subtract(img, open)
        # Step 4: Erode the original image and refine the skeleton
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        if cv2.countNonZero(img) == 0:
            break

    img = cv2.bitwise_not(skel)

    winname = "Garbor" + str(n)
    winsk = "Skeleton" + str(n)
    winoroginal = "Original" + str(n)
    cv2.namedWindow(winname)  # Create a named window
    cv2.namedWindow(winoroginal)  # Create a named window
    cv2.namedWindow(winsk)  # Create a named window
    cv2.moveWindow(winname, x, 0)  # Move it to (100,500)
    cv2.moveWindow(winoroginal, x, 200)  # Move it to (100,500)
    cv2.moveWindow(winsk, x, 400)  # Move it to (100,500)
    cv2.namedWindow(winname, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(winoroginal, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(winsk, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(winoroginal, cv2.resize(imgOriginal,(100,100),None,None,cv2.INTER_AREA))
    cv2.imshow(winname, cv2.resize(res1,(100,100),None,None,cv2.INTER_AREA))
    cv2.imshow(winsk, cv2.resize(img,(100,100),None,None,cv2.INTER_AREA))
    x += 100

    n += 1
    gravaImagem(res1,garbo+"garbo" + str(n) + ".jpg")
    gravaImagem(img, skeleton + "skeletonized" + str(n) + ".jpg")

cv2.waitKey(0)
cv2.destroyAllWindows()

