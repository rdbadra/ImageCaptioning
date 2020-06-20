# ImageCaptioning

Este repositorio forma parte del Trabajo de Fin de Máster del Máster en Ciencia de Datos de la Universitat Oberta de Catalunya. El objetivo de este trabajo es entrenar un modelo neuronal para generar descripciones de imágenes. El modelo está basado en una architectura Encoder-Decoder, siendo una CNN, la primiera, para convertir las imágenes en un vector de características, y el segundo una LSTM para generar descripciones a partir de este vector.

## Estructura

## Clonar el repositorio

```bash
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI/
make
python setup.py build
python setup.py install
cd ../../
git clone https://github.com/rdbadra/ImageCaptioning.git
cd ImageCaptioning/
```

## Descarga de datos

```bash
chmod +x download.sh
./download.sh
```

## Entrenamiento

```bash
python train.py
```

## Predicción

```bash
python predict.py --image='png/clock.png'
```


