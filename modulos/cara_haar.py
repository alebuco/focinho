import cv2
import os
import glob
import detecçao_focinho_haar

classificador = cv2.CascadeClassifier("cachorro_cara.xml")

for imagem in glob.glob(os.path.join("amostras", "*.jpg")):
    img = cv2.imread(imagem)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    focinhos_detectados = classificador.detectMultiScale(img_gray, scaleFactor=3, minSize=(190, 190))
    for focinho in focinhos_detectados:
        x, y, l, a = focinho[0], focinho[1], focinho[2], focinho[3]
        cv2.rectangle(img, (x, y), (x + l, y + a), (0, 255, 0), 2)
    winname = "Test"
    cv2.namedWindow(winname)  # Create a named window
    cv2.moveWindow(winname, 40, 30)  # Move it to (40,30)
    cv2.namedWindow(winname, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(winname, detecçao_focinho_haar.resize(img,50))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
