import cv2
import time
import os
import matplotlib as plt
import numpy as np
from translate import Translator as tl
import pyttsx3
print(cv2.__version__)
import RPi.GPIO as GPIO

botao_pin = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(botao_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

fps = 1
translator = tl(to_lang="pt-BR")

engine = pyttsx3.init()
engine.setProperty('voice', 'brazil')
rate = engine.getProperty('rate')
engine.setProperty('rate', rate-40)

labels_path = os.path.sep.join(['cfg/coco.names'])
LABELS = open(labels_path).read().strip().split('\n')

weights_path = os.path.sep.join(['./yolov4.weights'])
config_path = os.path.sep.join(['cfg/yolov4.cfg'])
net = cv2.dnn.readNet(config_path, weights_path)

COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype='uint8')
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

def falar(texto):
  engine.say(texto)
  print(texto)
  engine.runAndWait()
  engine.stop()

def mostrar(img):
  fig = plt.gcf()
  fig.set_size_inches(8,6)
  plt.axis('off')
  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  plt.show()

# mostrar(video)

def blob_imagem(net, video, mostrar_texto=True):
  inicio = time.time()

  blob = cv2.dnn.blobFromImage(video, 1 / 255.0, (320, 320), swapRB=True, crop=False)
  net.setInput(blob)
  layerOutputs = net.forward(ln)

  termino = time.time()
  print(f'\nYOLO levou {(termino-inicio):.2f} segundos')

  return net, video, layerOutputs

def deteccoes(detection, _threshold, caixas, confiancas, IDclasses):
    scores = detection[5:]
    classeID = np.argmax(scores)
    confianca = scores[classeID]

    if confianca > _threshold:
      caixa = detection[0:4] * np.array([W, H, W, H])
      (centerX, centerY, width, height) = caixa.astype('int')
      x = int(centerX - (width / 2))
      y = int(centerY - (height / 2))

      caixas.append([x, y, int(width), int(height)])
      confiancas.append(float(confianca))
      IDclasses.append(classeID)

    return caixas, confiancas, IDclasses

def funcoes_imagem(imagem, i, confiancas, caixas, COLORS, LABELS, mostrar_texto=True):  
  (x, y) = (caixas[i][0], caixas[i][1])
  (w, h) = (caixas[i][2], caixas[i][3])

  cor = [int(c) for c in COLORS[IDclasses[i]]]
  cv2.rectangle(imagem, (x, y), (x + w, y + h), cor, 2) 
  
  objeto = LABELS[IDclasses[i]]
  texto_traduzido = translator.translate(objeto)

  texto = f"{texto_traduzido}: {confiancas[i]:.4f}"
  if mostrar_texto:
    print("\n> " + texto)
    print(x,y,w,h)
  cv2.putText(imagem, texto, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

  return imagem,x,y,w,h,texto_traduzido


# Captura de vídeo com o OpenCV
print("\n*** Processamento do vídeo ***")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, fps)

try:
  while True:
    if GPIO.input(botao_pin) == GPIO.LOW:
    
      conectado, frame = cap.read()
      frame_largura = frame.shape[1]
      frame_altura = frame.shape[0]

      _threshold = 0.5
      _threshold_NMS = 0.3
      fonte_pequena, fonte_media = 0.4, 0.6
      fonte = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
      amostras_exibir = 20
      amostra_atual = 0

      t = time.time()
      frame = cv2.resize(frame, (frame_largura, frame_altura))
      frame = cv2.flip(frame,-1)
      
      try:
        (H,W) = frame.shape[:2]
      except:
        print("Erro")
        continue

      frame_cp = frame.copy()
      net, frame, layerOutputs = blob_imagem(net, frame)
      caixas = []       
      confiancas = []   
      IDclasses = []

      for output in layerOutputs:
        for detection in output:
          caixas, confiancas, IDclasses = deteccoes(detection, _threshold, caixas, confiancas, IDclasses)

      objs = cv2.dnn.NMSBoxes(caixas, confiancas, _threshold, _threshold_NMS)

      if len(objs) > 0:
        for i in objs.flatten():
          frame, x, y, w, h, texto_traduzido = funcoes_imagem(frame_cp, i, confiancas, caixas, COLORS, LABELS)

          # Calcula o centro da bounding box
          centerX = x + (w / 2)
          centerY = y + (h / 2)

          (H, W) = frame.shape[:2]

          # Calcula o centro da imagem
          centerImageX = W / 2
          centerImageY = H / 2

          # Calcular a posição relativa
          posicao_relativa = centerX - centerImageX

          # # InterprEstou vendoetar a posição relativa
          # if posicao_relativa < 0:
          #     print("O objeto está à esquerda.")
          # elif posicao_relativa > 0:
          #     print("O objeto está à direita.") 
          # else:
          #     print("O objeto está centralizado.")
        
          # Calcular a posição relativa
          posicao_relativa_x = centerX - centerImageX
          posicao_relativa_y = centerY - centerImageY
            
          # Calcular a porcentagem em relação à largura e altura da imagem
          percentagem_x = (abs(posicao_relativa_x) / (W / 2)) * 100
          percentagem_y = (abs(posicao_relativa_y) / (H / 2)) * 100
          
          #Interpretar a posição relativa
          if posicao_relativa_x < 0:
              falar("Estou vendo {} à esquerda.".format(texto_traduzido))
          elif posicao_relativa_x > 0:
              falar("Estou vendo {} à direita.".format(texto_traduzido))
          else:
             falar("Estou vendo {} centralizado.".format(texto_traduzido))

          #if posicao_relativa_y < 0:
          #    falar("Uma {} está acima".format(texto_traduzido))
          #elif posicao_relativa_y > 0:
          #    falar("Uma {} está abaixo".format(texto_traduzido))
          #else:
          #    falar("Uma {} está centralizado verticalmente.".format(texto_traduzido))
    


          cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0) , 2)    

          objeto = frame_cp[y:y + h, x:x + w]
          # falar(texto_traduzido)

        cv2.putText(frame, f"FPS: {cap.get(cv2.CAP_PROP_FPS)}", (20, H-20), fonte, fonte_pequena, (250, 250, 250, 0), lineType=cv2.LINE_AA)
        cv2.imshow("frame", frame)

      if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        
    else: 
      print("Botão não pressionado        ")
    time.sleep(0.1)

except keyboardInterrupt:
    GPIO.cleanup()
    
cap.release()
cv2.destroyAllWindows()

