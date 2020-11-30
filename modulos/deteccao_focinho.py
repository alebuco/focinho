import cv2
import dlib

detector_de_focinho_svm = dlib.simple_object_detector("../biblioteca/detector_focinhos.svm")
pontosFocinho = dlib.shape_predictor("../biblioteca/detector_focinhos_36_pontos_ccl.dat")
classificador = cv2.CascadeClassifier("../biblioteca/cachorro_cara.xml")
svm = dlib.simple_object_detector("../biblioteca/detector_focinhos.svm")
detectorCaraCachorro = dlib.simple_object_detector("../biblioteca/cara_cachorro.svm")
detectorPontosCachorro = dlib.shape_predictor("../biblioteca/detector_cara_cachorro_ccl.dat")
focinhos_haar = cv2.CascadeClassifier("../biblioteca/reconhecimento_focinhos_haar.xml")

def resize(img, percentual):
    scale_percent = percentual  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return resized

def imprimirPontos (imagem, pontos):
    for p in pontos.parts():
        cv2.circle(imagem, (p.x, p.y), 2, (0,255,0))

def focinho_haar(imagem,sf,mx,my):

    # conversão da imagem rgb em cinza
    img_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # detecção dos cachorros pelo haar cascade
    cachorros_detectados = classificador.detectMultiScale(img_gray, scaleFactor= sf, minSize=(mx, my))

    focinhoidentificado= None
    # identifica se detectou algum cachorro na imagem
    if cachorros_detectados is not None:

        # quantifica as possíveis imagens detectadas
        numero = len(cachorros_detectados)

        # percorre cada detecção do haar cascade
        for cachorro in cachorros_detectados:

            # coordenadas das caixas haar cascade
            x, y, l, a = cachorro[0], cachorro[1], cachorro[2], cachorro[3]

            # recorta as caixas haar
            focinhorecortado = imagem[y:y+a,x:x+l]

            # procura dentro da caixa selecionada, reconhecer o focinho pelo SVM
            focinhoidentificado = reconhece_focinho_svm(focinhorecortado, 2)

            print("Numero do teste realizado " + str(numero))
            numero -= 1

            if focinhoidentificado is not None:
                print("Achei focinho svm")

            else:
                print("Não foram encontrados os pontos")

    else:
        print ("HAAR nao encontrou cachorro")

    return focinhoidentificado

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
            # cv2.imshow("Imagem carregada no svm", imagem)

            # recorta a imagem do focinho identificado pelo svm
            # focinhorecortado = imagem[y:y+a,x:x+l]
            focinhorecortado = imagem[t:b, e:d]
            print("Focinho encontrado")

            identifica_pontos(imagem,focinho)

            winname = "SVM"
            cv2.namedWindow(winname)  # Create a named window
            cv2.moveWindow(winname,500, 0)  # Move it to (500, 0)
            cv2.namedWindow(winname, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(winname, imagem)
            cv2.waitKey(0)
            cv2.destroyWindow(winname)
    else:
        print("SVM não está achando")

    return focinhorecortado

def identifica_pontos(imagem,img):
    focinhopontos = pontosFocinho(imagem, img)
    # condição que verifica se existe algum arquivo onde foram detectados os pontos
    if focinhopontos is not None:

        # identifica os pontos nos focinhos localizados no svm
        for p in focinhopontos.parts():
            # mapeamento dos pontos de referencia no focinho detectado
            cv2.circle(imagem, (p.x, p.y), 2, (0, 255, 255))

    else:
        print("região não é focinho")

    cv2.imshow("Pontos do focinho", imagem)
    return imagem

def detector_cara(imagem, n):
    amostra = 1
    caes = detectorCaraCachorro(imagem, n)
    focinho = None
    if caes is not None:
        for caras in caes:
            e, t, d, b = (int(caras.left()), int(caras.top()), int(caras.right()), int(caras.bottom()))
            cv2.rectangle(imagem, (e,t), (d,b), (0,0,255), 3)

            # recorta caixa de deteccao da cara
            cara = imagem[t:b, e:d]

            # pontos = detectorPontosCachorro(imagem, caras)

            focinho = reconhece_focinho_svm(cara, 1)
            if focinho is not None:
                cv2.imshow("Crop",focinho)

            # cv2.waitKey(0)

            else:
                # cara = cv2.cvtColor(cara,cv2.COLOR_BGR2GRAY)
                focinho = detecta_focinho_haar(cara,1.8)

    else:
        print("Não houve detecção")

    cv2.imshow("Frontal ",imagem)
    cv2.waitKey(0)

    return focinho

def detecta_focinho_haar (imagem, i):

    # faz a verificação dos boxes marcados no haar cascade pelo svm
    haar = focinho_haar(imagem, i, 100, 100)
    focinhorecortado = None

    # verifica se ha focinhos detectados pelo haar na imagem
    if haar is not None:

        # percorre cada focinho confirmado pelo haar
        for focinho in haar:
            focinho_identificado = None

            # coordenadas dos focinhos haar detectados
            e, t, d, b = (int(focinho.left()), int(focinho.top()), int(focinho.right()), int(focinho.bottom()))
            cv2.rectangle(imagem, (e, t), (d, b), (255, 255, 0), 2)
            # cv2.imshow("Imagem carregada no haar", imagem)

            # recorta a imagem do focinho identificado pelo haar
            # focinhorecortado = imagem[y:y+a,x:x+l]
            focinhorecortado = imagem[t:b, e:d]
            print("Focinho encontrado")

            # redimensiona o focinho recortado em imagem 100 x 100
            # focinhorecortado = cv2.resize(focinhorecortado,(100,100),interpolation=cv2.INTER_AREA)

            pontos = identifica_pontos(imagem,focinho)
            if pontos is None:
                print("Haar: não detectou pontos")

            winname = "HAAR"
            cv2.namedWindow(winname)  # Create a named window
            cv2.moveWindow(winname,100, 10)  # Move it to (100, 10)
            cv2.namedWindow(winname, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(winname, imagem)
            cv2.waitKey(0)
            cv2.destroyWindow(winname)
    else:
        print("HAAR não está achando")

    return focinhorecortado
