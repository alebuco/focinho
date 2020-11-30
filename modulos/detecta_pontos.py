import cv2
import dlib

pontosFocinho = dlib.shape_predictor("../biblioteca/detector_focinhos_36_pontos_ccl.dat")

def identifica_pontos(imagem,img):
    focinhopontos = pontosFocinho(imagem, img)
    # condição que verifica se existe algum arquivo onde foram detectados os pontos
    if focinhopontos is not None:

        # identifica os pontos nos focinhos localizados no svm
        for p in focinhopontos.parts():
            # print("Coordenadas (X,Y) = "+ str(p.x)+ " , " + str(p.y))
            cv2.circle(imagem, (p.x, p.y), 2, (255, 255, 255))

        # x, y, a, l = focinhopontos[0], focinhopontos[1], focinhopontos[2], focinhopontos[3]
        # cv2.rectangle(imagem, (x,y), (x+a, y+l), (0, 0, 255), 4)

    else:
        print("região não é focinho")

    cv2.imshow("", imagem)
    return imagem
