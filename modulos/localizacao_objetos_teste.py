import dlib
import cv2

def recortarRetangulo (imagem, e,t,d,b):
    crop = imagem[t:b, e:d]

def imprimirPontos (imagem, pontos):
    for p in pontos.parts():
        cv2.circle(imagem, (p.x, p.y), 2, (0,255,0))

detectorFocinho = dlib.simple_object_detector("detector_focinhos.svm")
detectorPontosFocinho = dlib.shape_predictor("positivas_manual/detector_focinhos_36_pontos_ccl.dat")

image_file = 'amostras/rede/caes.1.jpg'
img = dlib.load_rgb_image(image_file)

rects = []
dlib.find_candidate_object_locations(img, rects, min_size=500)
k = len(rects)
print("number of rectangules found {}".format(k))
for d in rects:
    e, t, d, b = int(d.left()), int(d.top()), int(d.right()), int(d.bottom())
    print("Detection {}; Left: {} Top: {} Right: {} Botton: {}". format(k,e,t,d,b))
    cv2.rectangle(img, (e,t), (d,b), (0,0,255), 2)
    recorte = recortarRetangulo(img, e,t,d,b)
    objetosDetectados = detectorFocinho(recorte, 2)
    for cara in objetosDetectados:
        e, t, d, b = (int(cara.left()), int(cara.top()), int(cara.right()), int(cara.bottom()))
        cv2.rectangle(imagem, (e,t), (d,b), (0,0,255), 2)
        pontos = detectorPontosFocinho(imagem, cara)
        imprimirPontos(imagem, pontos)


cv2.imshow("Possiveis objetos", img)
cv2.waitKey(0)
cv2.destroyAllWindows()