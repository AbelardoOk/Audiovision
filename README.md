# Programa para detectar objetos em imagens em raspberry com uso de Webcam

## Resumo
Programa desenvolvido para um projeto de extensão da FundectMS, feito por estudantes do IFMS Campus Aquidauana e orientado pelo professor Ygo Brito e Vinícius Maeda.

Utilizando uma webcam, o programa irá realizar a detecção a partir do código previamente feito, no qual irá capturar a imagem, realizar as detecções dos objetos e retornar ao usuário os resultados em forma de áudio.

## Versões das bibliotecas

- OpenCV: 4.9.0
- Pyttsx3: 2.90
- Translate: 3.6.1
- Ultralytics: 8.2.19
- RPi.GPIO: 0.7.1 

## Instalação/Preparação do ambiente

#### Importante
Para o funcionamento do código, é necessário que seja rodado em uma raspberry com um push button conectado ao pino 17

### Iniciar:
Após os arquivos serem clonados e instalados corretamente, digite:
`python program.py`

## Resultado esperado
Ao executar o programa, espera-se que abra um janela com a imagem da webcam utilizada na detecção com os objetos reconhecidos demarcados e que também seja narrado o nome do mesmo.
