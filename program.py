from ultralytics import YOLO
import cv2
from translate import Translator as tl
import pyttsx3
import RPi.GPIO as GPIO
import time
import os

# Configuração botão
botao_pin = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(botao_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Código utilizando o modelo YoloV8 Medium
model = YOLO('yolov8n.pt')

# Configurações de fala
translator = tl(to_lang="pt-BR")

engine = pyttsx3.init()
engine.setProperty('voice', 'brazil')
rate = engine.getProperty('rate')
engine.setProperty('rate', rate-40)

def falar(texto):
    engine.say(f"Estou vento {texto}")
    engine.runAndWait()
    engine.stop()


# Configuração foto
comandoFoto = "ffmpeg -f video4linux2 -s 640x480 -ss 4 -i /dev/video0 -frames 1 ./foto.jpg"

# Processamento foto
try:
  while True:
      if GPIO.input(botao_pin) == GPIO.LOW:
          os.system("rm foto.jpg")
          os.system(comandoFoto)
        
          img = cv2.imread("foto.jpg")
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
                  objeto = translator.translate(classNames[cls])
                  objetos.append(objeto)

          print(f"Objetos detectados: {objetos    }")
          falar(objetos)

      else:
          print("Botão não prssionado")

      if cv2.waitKey(1) & 0xFF == ord("q"):
        break
      time.sleep(0.1)

except KeyboardInterrupt:
    GPIO.cleanup()

cv2.destroyAllWindows()
