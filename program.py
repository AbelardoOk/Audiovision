from ultralytics import YOLO
import cv2
import pyttsx3
import RPi.GPIO as GPIO
import time
import os
import numpy as np
#import translate as tl

#from_code = "en"
#to_code = "pt"

# Configuração botão
botao_pin = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(botao_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Código utilizando o modelo YoloV8 Medium
model = YOLO('yolov8n.pt')

# Configurações de fala
#translator = tl(to_lang="pt-BR")

engine = pyttsx3.init()
engine.setProperty('voice', 'brazil')
rate = engine.getProperty('rate')
engine.setProperty('rate', rate-40)

def falar(texto):
    engine.say(f"Estou vento {texto}")
    engine.runAndWait()
    engine.stop()


# Configuração foto
cap = cv2.VideoCapture(0)

def ajustar_balanco_branco(imagem):
    resultado = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB)
    avg_a = np.average(resultado[:, :, 1])
    avg_b = np.average(resultado[:, :, 2])
    resultado[:, :, 1] = resultado[:, :, 1] - ((avg_a - 128) * (resultado[:, :, 0] / 255.0) * 1.1)
    resultado[:, :, 2] = resultado[:, :, 2] - ((avg_b - 128) * (resultado[:, :, 0] / 255.0) * 1.1)
    return cv2.cvtColor(resultado, cv2.COLOR_LAB2BGR)
    
def iniciar_camera():
    cap = cv2.VideoCapture(0)  # Tente abrir a câmera (0 para a webcam padrão)
    if not cap.isOpened():
        print("Erro ao acessar a câmera.")
    return cap

def tirarFoto(cap):
    time.sleep(1)
    for i in range(30):
       ret, frame = cap.read()
       frame = ajustar_balanco_branco(frame)
    return frame

print("Pressione o botão para começar a detectar")

# Processamento foto
try:
  while True:
      if not cap.isOpened():
        cap = iniciar_camera()
        
      if GPIO.input(botao_pin) == GPIO.LOW:
          img = tirarFoto(cap)
          resultado = model(img, stream=True)
          
          img = cv2.flip(img,-1)
          cv2.imshow('Foto', img)

          for r in resultado:
              print("\n")
              boxes = r.boxes
              classNames = r.names
              objetos = []

              for box in boxes:
                  # Classe detectada
                  cls = int(box.cls[0])
                  objeto = classNames[cls]
                  #objeto = translator.translate(classNames[cls])
                  objetos.append(objeto)

          print(f"Objetos detectados: {objetos}")
          falar(objetos)
          print("\nPressione o botão para começar a detectar")
          cap.release()

      #else:
          #print("Botão não prssionado")

      if cv2.waitKey(1) & 0xFF == ord("q"):
        break
      time.sleep(0.1)

except KeyboardInterrupt:
    GPIO.cleanup()

cv2.destroyAllWindows()
