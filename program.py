# Importação das bibliotecas no código
from ultralytics import YOLO
import cv2
import pyttsx3
import RPi.GPIO as GPIO
import time
import numpy as np
from translate import Translator as tl

# Configuração do botão usando GPIO (pino 17)
botao_pin = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(botao_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Código utilizando o modelo YoloV8 Medium
model = YOLO('yolov8n.pt')
model.export(format='ncnn')
ncnn_model = YOLO("./yolo11n_ncnn_model")

# Configurações de fala
translator = tl(to_lang="pt-BR")

# Configurações de tradução e fala
engine = pyttsx3.init()
engine.setProperty('voice', 'brazil')
rate = engine.getProperty('rate')
engine.setProperty('rate', rate-40)

# Função para falar texto usando o mecanismo de fala
def falar(texto):
    engine.say(f"Estou vendo {texto}")
    engine.runAndWait()
    engine.stop()

# Configuração inicial da câmera
cap = cv2.VideoCapture(1)

# Função para inicializar a câmera caso ocorra falha                                   
def iniciar_camera():
    cap = cv2.VideoCapture(1)  # Tente abrir a câmera (0 para a webcam padrão)
    if not cap.isOpened():
        print("Erro ao acessar a câmera.")
    return cap

print("Pressione o botão para começar a detectar")

# Loop principal para processamento de imagens
try:
  while True:
      if not cap.isOpened():
        print("Pressione o botão")
        cap = iniciar_camera()
        
      ret, frame = cap.read()
    
      # Verifica se o botão foi pressionado
      if GPIO.input(botao_pin) == GPIO.LOW:
          img = frame
          img = cv2.flip(img,-1)
          resultado = ncnn_model(img, stream=True)        
          cv2.imshow('Foto', img)

          # Processa cada resultado da detecção
          for r in resultado:
              print("\n")
              boxes = r.boxes
              classNames = r.names
              objetos = []

              for box in boxes:
                  # Identifica a classe detectada e traduz para português
                  cls = int(box.cls[0])
                  objeto = classNames[cls]
                  objeto = translator.translate(classNames[cls])
                  objetos.append(objeto)

          print(f"Objetos detectados: {objetos}")
          falar(objetos)
          cap.release()

      # Finaliza o programa se a tecla 'q' for pressionada
      if cv2.waitKey(1) & 0xFF == ord("q"):
        break
      time.sleep(0.1)

except KeyboardInterrupt:
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()
