import cv2
import dlib

classificador = cv2.CascadeClassifier("../biblioteca/reconhecimento_focinhos_haar.xml")
pontosFocinho = dlib.shape_predictor("../biblioteca/detector_focinhos_36_pontos_ccl.dat")
detector_de_focinho_svm = dlib.simple_object_detector("../biblioteca/detector_focinhos.svm")

def resize(img, percentual):
    scale_percent = percentual  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return resized

def focinho_haar(imagem,sf,mx,my):

    # conversão da imagem rgb em cinza
    img_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # detecção dos focinhos pelo haar cascade
    focinhos_detectados = classificador.detectMultiScale(img_gray, scaleFactor= sf, minSize=(mx, my))
    focinhorecortado100x100 = None
    if focinhos_detectados is not None:
        focinhorecortado = None
        numero = len(focinhos_detectados)

        # percorre cada detecção do haar cascade
        for fuco in focinhos_detectados:

            # coordenadas das caixas haar cascade
            x, y, l, a = fuco[0], fuco[1], fuco[2], fuco[3]
            # cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 255, 0), 2)

            # recorta as caixas haar
            crop_f = imagem[y:y+a,x:x+l]

            # faz a verificação dos boxes marcados no haar cascade pelo svm
            f_svm = detector_de_focinho_svm(crop_f, 4)

            # verifica se os boxes marcados pelo haar correspondem a focinhos detectados pelo svm
            if f_svm is not None:

                # percorre cada focinho confirmado pelo svm
                for focinho in f_svm:

                    # coordenadas dos focinhos svm detectados
                    e, t, d, b = (int(focinho.left()), int(focinho.top()), int(focinho.right()), int(focinho.bottom()))

                    # recorta a imagem do focinho identificado pelo svm
                    focinhorecortado = imagem[y:y+a,x:x+l]

                    # redimensiona o focinho recortado em imagem 100 x 100
                    focinhorecortado100x100 = cv2.resize(focinhorecortado,(100,100),interpolation=cv2.INTER_AREA)
                    # cv2.imshow(" ",focinhorecortado)
                    focinhopontos = pontosFocinho(imagem, focinho)

                    # condição que verifica qe existe algum arquivo onde foram detectados os pontos
                    if focinhopontos is not None:

                        # identifica os pontos nos focinhos localizados no svm
                        for p in focinhopontos.parts():
                            print("Coordenadas (X,Y) = "+ str(p.x)+ " , " + str(p.y))
                            cv2.circle(focinhorecortado, (p.x, p.y), 4, (255, 255, 255))

                        focinhorecortado = focinhorecortado[30:70, 30:70]
                        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 4)
                        # focinhorecortadox200 = resize(focinhorecortado,200)
                        cv2.waitKey(0)
                    else:
                        print("região não é focinho")
                        return focinhorecortado100x100

            print("Numero do teste realizado " + str(numero))
            numero -= 1
            # cv2.waitKey(0)

        winname = "Imagem"
        cv2.namedWindow(winname)  # Create a named window
        cv2.moveWindow(winname,500, 10)  # Move it to (500,30)
        cv2.namedWindow(winname, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(winname, resize(imagem,50))
        cv2.waitKey(0)
    else:
        print ("HAAR nao funciona")
    return focinhorecortado100x100
