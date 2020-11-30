import dlib
import cv2
import glob
import os

detectorFocinho = dlib.simple_object_detector("detector_focinhos.svm")
detectorPontosFocinho = dlib.shape_predictor("positivas_manual/detector_focinhos_36_pontos_ccl.dat")

def imprimirPontos (imagem, pontos):
    for p in pontos.parts():
        cv2.circle(imagem, (p.x, p.y), 2, (0,255,0))


# print(dlib.test_shape_predictor("recursos/teste_relogios_pontos.xml", "recursos/detector_relogios_pontos.dat"))

for arquivo in glob.glob(os.path.join("focinhos_cachorro", "*.jpg")):
    imagem = dlib.load_rgb_image(arquivo)
    imagem_grayscale = dlib.as_grayscale(imagem)
    objetosDetectados = detectorFocinho(imagem, 2)
    for cara in objetosDetectados:
        e, t, d, b = (int(cara.left()), int(cara.top()), int(cara.right()), int(cara.bottom()))
        cv2.rectangle(imagem, (e,t), (d,b), (0,0,255), 2)
        pontos = detectorPontosFocinho(imagem, cara)
        imprimirPontos(imagem, pontos)

    cv2.imshow("Detector pontos",imagem)
    cv2.waitKey(0)

cv2.destroyAllWindows()