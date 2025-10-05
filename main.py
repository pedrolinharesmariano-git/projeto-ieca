# Importando os pacotes necessários
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import matplotlib.pyplot as plt

# Função para calcular a relação de aspecto dos olhos (EAR)
def calcular_ear(olho):
    A = dist.euclidean(olho[1], olho[5])
    B = dist.euclidean(olho[2], olho[4])
    C = dist.euclidean(olho[0], olho[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Constroi o parser dos argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="índice da webcam no sistema")
args = vars(ap.parse_args())

# <<< MUDANÇAS AQUI >>>
# Defina o limiar do EAR e o número de quadros consecutivos
# que o olho precisa estar abaixo do limiar para ser considerado "fechado"
EAR_THRESH = 0.25
EAR_CONSEC_FRAMES = 10 # Aumente este valor para ser menos sensível

# Inicializa o contador de quadros consecutivos e o status dos olhos
CONTADOR = 0
# <<< FIM DAS MUDANÇAS >>>


# Inicializa o detector de rosto do dlib (baseado em HOG) e cria
# o preditor de marcos faciais
print("[INFO] Carregando preditor de marcos faciais...")
detector = dlib.get_frontal_face_detector()
preditor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Pega os índices dos marcos faciais para o olho esquerdo e direito
(inicio_esq, fim_esq) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(inicio_dir, fim_dir) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Inicia a thread de fluxo de vídeo
print("[INFO] Iniciando thread de fluxo de vídeo...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

# (Opcional) Configuração do Matplotlib
y = [None] * 100
x = np.arange(0,100)
fig = plt.figure()
ax = fig.add_subplot(111)
li, = ax.plot(x, y)
plt.xlim([0, 100])
plt.ylim([0, 0.5])
plt.title("EAR ao longo do tempo")
plt.xlabel("Quadros")
plt.ylabel("EAR")


# Loop nos quadros do vídeo
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=700)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = preditor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        olho_esq = shape[inicio_esq:fim_esq]
        olho_dir = shape[inicio_dir:fim_dir]
        ear_esq = calcular_ear(olho_esq)
        ear_dir = calcular_ear(olho_dir)

        ear = (ear_esq + ear_dir) / 2.0

        casco_olho_esq = cv2.convexHull(olho_esq)
        casco_olho_dir = cv2.convexHull(olho_dir)
        cv2.drawContours(frame, [casco_olho_esq], -1, (255, 255, 0), 1)
        cv2.drawContours(frame, [casco_olho_dir], -1, (255, 255, 0), 1)

        # <<< LÓGICA MELHORADA AQUI >>>
        # Verifica se o EAR está abaixo do limiar
        if ear < EAR_THRESH:
            CONTADOR += 1

            # Se os olhos ficaram fechados por um número suficiente de quadros,
            # então exiba a mensagem
            if CONTADOR >= EAR_CONSEC_FRAMES:
                texto_status = "OLHOS FECHADOS"
                cor_status = (0, 0, 255) # Vermelho
            else:
                # Caso contrário, ainda consideramos aberto, mas em alerta
                texto_status = "OLHOS ABERTOS"
                cor_status = (0, 255, 0) # Verde

        else:
            # Se o EAR não está abaixo do limiar, reseta o contador
            CONTADOR = 0
            texto_status = "OLHOS ABERTOS"
            cor_status = (0, 255, 0) # Verde

        # Desenha o status na tela
        cv2.putText(frame, texto_status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor_status, 2)
        # <<< FIM DA LÓGICA MELHORADA >>>

        # (Opcional) Atualiza o gráfico do Matplotlib
        y.pop(0)
        y.append(ear)
        li.set_ydata(y)
        fig.canvas.draw()
        plt.pause(0.01)

        # Exibe o valor do EAR para ajudar a calibrar o limiar
        cv2.putText(frame, "EAR: {:.3f}".format(ear), (500, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 150), 2)

    cv2.imshow("Detecção de Olhos", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
plt.close('all')