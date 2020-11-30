import dlib
import cv2

detectorFocinho = dlib.simple_object_detector("detector_focinhos.svm")
pontosFocinho = dlib.shape_predictor("focinhos_cachorro/detector_focinhos_15_pontos.dat")



def imprimirPontos(imagem, pontos):
    for p in pontos.parts():
        cv2.circle(imagem, (p.x, p.y), 2, (0, 255, 0))


def recortaFocinho(imagem, e, t, d, b):
    crop = imagem[t:b, e:d]
    resized_image = cv2.resize(crop, (100, 100))
    # negativa = negative_image.negativa(cv2.cvtColor(crop,cv2.IMREAD_GRAYSCALE))
    cv2.imshow("Rct", resized_image)
    cv2.waitKey(0)
    return resized_image

def limita_focinho(imagem,n):
    focinhoDetectado = detectorFocinho(imagem, n)
    focinhorecortado = None
    print("Localizando o focinho")
    try:
        for focinho in focinhoDetectado:
            e, t, d, b = (int(focinho.left()), int(focinho.top()), int(focinho.right()), int(focinho.bottom()))
            focinhorecortado = recortaFocinho(focinho, e, t, d, b)
            cv2.imshow("Recortando com svm",focinhorecortado)
            cv2.rectangle(imagem, (e, t), (d, b), (0, 0, 255), 2)
                    # pontos = pontosFocinho(imagem, cachorro)
                    # imprimirPontos(imagem, pontos)
            print("Focinho detectado")
            cv2.waitKey(0)

    except Exception as erro :
        print("Erro", erro)
    return focinhorecortado
