from array import ArrayType
import keras
import librosa
import librosa.display 
import numpy as np
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt


FILE_PATH = "Predict\Piano_F4_1657927716.9162831.wav"
DATASET_PATH = "Data"
JSON_PATH = "data_chord.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 3 # measured in seconds
n_fft = 2048
hop_length = 512
CATEGORIES = ["A#3","A#4","A#5","A3","A4","A5",
              "B3", "B4","B5","C#3","C#4","C#5",
              "C3","C4","C5","D#3","D#4", "D#5",
              "D3","D4","D5","E3","E4", "E5",
              "F#3","F#4", "F#5","F3","F4","F5",
              "G#3","G#4","G#5","G3","G4","G5",]

def correctShape(chroma_shape):
    return chroma_shape == 130

def normalizeShape(chroma_mat):
    nums = 0
    #init_shape tiene la dimension de columnas. 
    init_shape= chroma_mat.shape[1]
    #Me fijo cuantas columnas faltan por rellenar
    nums = 130 - init_shape
    #itero nums copiando el anterior
    arreglo = np.array(chroma_mat[:,init_shape-1])
   
    i = 0
    while i < nums :
        chroma_mat= np.column_stack((chroma_mat,arreglo))  
        i = i +1 
    return chroma_mat

def load_data(data_path):
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data["chroma"])
    y = np.array(data["labels"])

    print("Data succesfully loaded!")

    return X, y

def getChordFromRNN(file_path, sample_rate):
    my_model = keras.models.load_model('modelo-acordes-v01.h5')

    signal, sr = librosa.load(file_path, sr=sample_rate)

    # extract Chroma
    chroma = librosa.feature.chroma_cens(y=signal, sr=sr)

    if not correctShape(chroma.shape[1]) :
        chroma = normalizeShape(chroma)

    chroma_reshape = tf.reshape(chroma, [ 1,12,130])
    my_prediction = my_model.predict(chroma_reshape)
   
    index = np.argmax(my_prediction)
    print("chord: " + CATEGORIES[index])

    return CATEGORIES[index]


def getNotesFromChords(chord):
    notas_string = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']	
    nota_2 =''
    nota_3 =''
    octaveIndex = 2
    secondNoteIndex = 3

    if len(chord) == 2:
        octaveIndex = 1
        secondNoteIndex = 4

	
    nota = chord[0]
    octava = int(chord[octaveIndex])
    indice = notas_string.index(nota)
    if ((indice + secondNoteIndex) / 12) >= 1 :
        print(indice + secondNoteIndex)
        nota_2 = notas_string[((indice + secondNoteIndex )%12)] + str(octava + 1)
    else:
        nota_2 = notas_string[(indice + secondNoteIndex)]  + str(octava)
    if ((indice + 7) / 12) >= 1 :
        nota_3 = notas_string[((indice + 7) %12)] + str(octava + 1)
    else:
        nota_3 = notas_string[(indice + 7)]  + str(octava)

    triada = [chord, nota_2, nota_3]
     	 	
    return triada
    

if __name__ == "__main__":
    chord = getChordFromRNN(FILE_PATH,SAMPLE_RATE)
    print(chord)
    print("Notes from Chord")
    print(getNotesFromChords(chord))
#plt.figure(figsize=(25, 10))
#librosa.display.specshow(chroma, 
#                         y_axis="chroma", 
#                         x_axis="time",
#                         sr=sample_rate)
#plt.colorbar(format="%+2.f")
#plt.show()