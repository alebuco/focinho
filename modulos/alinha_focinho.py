import dlib
import cv2
import glob
import os
import numpy as np

def imprimePontos( imagem, pontosFaciais):
    for p in pontosFaciais.parts():
        cv2.circle(imagem, (p.x, p.y), 2, (0,255,0),2)

detectorFocinho = dlib.simple_object_detector("cara_cachorro.svm")
detectorPontos = dlib.shape_predictor("focinhos_cachorro/detector_focinhos_5pontos.dat")
imagem = cv2.imread("focinhos_cachorro/f.92.jpg")
imagemRgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
cachorroDetectado = detectorFocinho(imagemRgb, 1)
facesPontos = dlib.full_object_detections()
for cachorro in glob.glob(os.path.join("focinhos_cachorro", "*.jpg")):
    for focinho in cachorroDetectado:
        pontos = detectorPontos(imagemRgb, focinho)
        facesPontos.append(pontos)
        imprimePontos(imagem, pontos)

    imagens = dlib.get_face_chips(imagemRgb, facesPontos)
    if imagens :
        for img in imagens:
            imagemBgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow("Imagem original", imagem)
            cv2.waitKey(0)
            cv2.imshow("Imagem alinhada", imagemBgr)
            cv2.waitKey(0)

# cv2.imshow("5 pontos", imagem)
# cv2.waitKey(0)
cv2.destroyAllWindows()