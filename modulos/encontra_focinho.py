import glob
import os
import cv2
import modulos.deteccao_focinho as deteccao

amostra = 0
# percorre um diretório com as amstras de imagens dos caes
for arquivo in glob.glob(os.path.join("../imagens/amostras", "*.jpg")):
    imagem = cv2.imread(arquivo)
    amostra += 1
    print ("Amostra numero " + str(amostra))

    # detecta o cao com haar
    focinho = deteccao.focinho_haar(imagem, 1.8, 200, 200)

    # condição que verifica que se o cao não foi detectado pelo Haar cascade
    # exista uma segunda detecção pelo SVM
    if focinho is None:
        # detecção do cao com svm
        focinho = deteccao.detector_cara(imagem, 2)

cv2.waitKey(0)
cv2.destroyAllWindows()
