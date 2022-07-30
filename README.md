# Chord_cnn
## intro
En este repositorio van a encontrar el dataset y el soft para entrenar la red neuronal que reconozca acordes mayores y menores de piano y guitarra. 

##  Parsear información
Ejecutar py parser_data.py. 
INPUT: En carpeta DATA (piano y guitarra). audios de tres segundos. Se extrae el chroma con librosa.
OUTPUT: Archivo json (data_chords.json)

## Entrenar y Testear a la red
Ejecutar py RNN.py
INPUT: data_chords.json
OUTPUT: modelo-entrenado.h5 y gráfica con el accuracy y testing. 

## Requirements
*   Tensorflow: 	
    *   pip install Tensorflow
*   Librosa: 	
    *   pip install Librosa
*   Scipy:
    *	pip install scipy
*   Numpy:
    *	pip install numpy
*   Matplotlib:	
    *   pip install matplotlib
*   Sklearn:
    *	pip install sklearn