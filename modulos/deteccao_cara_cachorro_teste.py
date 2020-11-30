import dlib
import cv2
import glob
import os
import modulos.identificador_focinho_pontos_teste as detector_pontos

detectorCaraCachorro = dlib.simple_object_detector("../biblioteca/cara_cachorro.svm")
detectorPontosCachorro = dlib.shape_predictor("../biblioteca/detector_cara_cachorro_ccl.dat")

def imprimirPontos (imagem, pontos):
    for p in pontos.parts():
        cv2.circle(imagem, (p.x, p.y), 2, (0,255,0))

def recortaCara (crop, e,t,d,b):

    cv2.imshow("Recortada", crop)
    return crop


def detector_cara(imagem, n):
    amostra = 1
    caes = detectorCaraCachorro(imagem, n)
    try:
        for caras in caes:
            e, t, d, b = (int(caras.left()), int(caras.top()), int(caras.right()), int(caras.bottom()))
            cv2.rectangle(imagem, (e,t), (d,b), (0,0,255), 3)
            # crop = recortaCara(caras, e, t, d, b)
            crop = imagem[t:b, e:d]
            pontos = detectorPontosCachorro(imagem, caras)
            cv2.imshow("Crop",crop)
            imprimirPontos(imagem, pontos)
            print ("amostra " + str(amostra) + "\n")
            cv2.waitKey(0)
            amostra += 1
            focinho = detector_pontos.limita_focinho(crop, 3)
            cv2.waitKey(0)
    except Exception as e :
        print("Não houve detecção")

    cv2.imshow("Detector pontos",imagem)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return imagem
