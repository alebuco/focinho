import dlib
import cv2
import glob
import os

detectorFocinho = dlib.simple_object_detector("detector_focinhos.svm")

def recortaCara (imagem, e,t,d,b):
    crop = imagem[t:b, e:d]
    cv2.imshow("Rct", crop)
    cv2.waitKey(0)

for imagem in glob.glob(os.path.join("amostras", "*.jpg")):
    img = cv2.imread(imagem)
    focinhos_detectados = detectorFocinho (img, 2)
    for focinho in focinhos_detectados:
        e, t, d, b = (int(focinho.left()), int(focinho.top()), int(focinho.right()), int(focinho.bottom()))
        cv2.rectangle(img, (e,t), (d,b), (0,0,255), 2)
       # recortaCara(img, e,t,d,b)
    cv2.imshow("Detector de focinhos", img)
    cv2.waitKey(0)


cv2.destroyAllWindows()