import dlib
import cv2
import glob
import os

detectorFocinho = dlib.simple_object_detector("../biblioteca/detector_focinhos.svm")
pontosFocinho = dlib.shape_predictor("../biblioteca/detector_focinhos_36_pontos_ccl.dat")

def imprimirPontos (imagem, pontos):
    for p in pontos.parts():
        cv2.circle(imagem, (p.x, p.y), 2, (0,255,0))

def recortaFocinho (imagem, e,t,d,b):
    crop = imagem[t:b, e:d]
    resized_image = cv2.resize(crop, (100, 100))
    cv2.imshow("Rct", resized_image)
    cv2.waitKey(0)


for arquivo in glob.glob(os.path.join("amostras", "*.jpg")):
    imagem = cv2.imread(arquivo)
    objetosDetectados = detectorFocinho(imagem, 2)
    try :
        for cachorro in objetosDetectados:
            e, t, d, b = (int(cachorro.left()), int(cachorro.top()), int(cachorro.right()), int(cachorro.bottom()))
            recortaFocinho(imagem,e,t,d,b)
            cv2.rectangle(imagem, (e,t), (d,b), (0,0,255), 2)
            pontos = pontosFocinho(imagem, cachorro)
            imprimirPontos(imagem, pontos)


    except e:
        print ("nada detectado")
    cv2.imshow("Detector pontos",imagem)
    cv2.waitKey(0)

cv2.destroyAllWindows()