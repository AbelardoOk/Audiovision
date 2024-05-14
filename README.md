# Porgrama para detectar objetos em imagens em raspberry com uso de Webcam

## Resumo
Programa desenvolvido para um projeto de extensão da FundectMS, feito por estudantes do IFMS Campus Aquidauana e orientado pelo professor Ygo Brito e Vinícius Maeda.

Utilizando uma webcam, o programa irá realizar a detecção a partir do código previamente feito, no qual irá capturar a imagem, realizar as detecções dos objetos e retornar ao usuário os resultados em forma de áudio.

![Fluxograma processamento](https://media.discordapp.net/attachments/774432392818589746/1239951052997136425/Captacao_da_imagem_1.png?ex=6644c9e8&is=66437868&hm=f9223975e22b0e34031c5a1f9cc8e713eef828964f39cd4172307e107d58ae3e&=&format=webp&quality=lossless&width=1280&height=320)

## Versões das bibliotecas

- OpenCV: 4.9.0
- Pyttsx3: 2.90
- Matplotlib: 3.8.3
- Translate: 3.6.1

## Instalação/Preparação do ambiente

### Faça o download deste arquivo de pesos e insira na raiz do projeto:
`wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights`

#### Importante
`Antes de iniciar o código, é necessário instalar as dependências necessárias.`

## Resultado esperado
Ao executar o programa, espera-se que abra um janela com a imagem da webcam utilizada na detecção com os objetos reconhecidos demarcados e que também seja narrado o nome do mesmo.
