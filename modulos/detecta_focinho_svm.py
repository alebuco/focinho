import cv2
import dlib
import modulos.detecta_pontos as detecta_pontos

detector_de_focinho_svm = dlib.simple_object_detector("../biblioteca/detector_focinhos.svm")
pontosFocinho = dlib.shape_predictor("../biblioteca/detector_focinhos_36_pontos_ccl.dat")

def reconhece_focinho_svm (imagem, i):

    # faz a verificação dos boxes marcados no haar cascade pelo svm
    svm = detector_de_focinho_svm(imagem, i)
    focinhorecortado = None

    # verifica se ha focinhos detectados pelo svm na imagem
    if svm is not None:

        # percorre cada focinho confirmado pelo svm
        for focinho in svm:
            focinho_identificado = None

            # coordenadas dos focinhos svm detectados
            e, t, d, b = (int(focinho.left()), int(focinho.top()), int(focinho.right()), int(focinho.bottom()))
            cv2.rectangle(imagem, (e, t), (d, b), (255, 255, 0), 2)
            cv2.imshow("Imagem carregada no svm", imagem)

            # recorta a imagem do focinho identificado pelo svm
            # focinhorecortado = imagem[y:y+a,x:x+l]
            focinhorecortado = imagem[t:b, e:d]
            print("Focinho encontrado")


            # redimensiona o focinho recortado em imagem 100 x 100
            # focinhorecortado = cv2.resize(focinhorecortado,(100,100),interpolation=cv2.INTER_AREA)

            detecta_pontos.identifica_pontos(imagem, focinho)

            winname = "SVM"
            cv2.namedWindow(winname)  # Create a named window
            cv2.moveWindow(winname,100, 10)  # Move it to (100, 10)
            cv2.namedWindow(winname, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(winname, imagem)
            cv2.waitKey(0)
            cv2.destroyWindow(winname)
    else:
        print("SVM não está achando")

    return focinhorecortado