from array import ArrayType
import keras
import librosa
import librosa.display 
import numpy as np
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys 
from scipy.signal import find_peaks
from scipy.fft import fft

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
WORKSPACE = os.path.dirname(FILE_PATH)

sys.path.insert(0, os.path.join(WORKSPACE, "imput_parser"))



FILE_PATH = "Predict\Bb_min_5_0.wav"
DATASET_PATH = "Data"
JSON_PATH = "data_chord.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 3 # measured in seconds
n_fft = 2048
hop_length = 512
"""CATEGORIES = ["A#3","A#4","A#5","A3","A4","A5",
              "B3", "B4","B5","C#3","C#4","C#5",
              "C3","C4","C5","D#3","D#4", "D#5",
              "D3","D4","D5","E3","E4", "E5",
              "F#3","F#4", "F#5","F3","F4","F5",
              "G#3","G#4","G#5","G3","G4","G5",]"""
CATEGORIES = ["A","A#","A#-","A-","B","B-","C","C#","C#-","C-","D","D#",
              "D#-","D-","E","E-","F","F#","F#-","F-","G","G#","G#-","G-"]

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

def getChordFromRNN(signal, sample_rate):
    my_model = keras.models.load_model('modelo-acordes-v02.h5')


    # extract Chroma
    chroma = librosa.feature.chroma_cens(y=signal, sr=sample_rate,fmin=130,n_octaves=2)

    if not correctShape(chroma.shape[1]) :
        chroma = normalizeShape(chroma)

    chroma_reshape = tf.reshape(chroma, [ 1,12,130])
    my_prediction = my_model.predict(chroma_reshape)
    print(my_prediction)
    index = np.argmax(my_prediction)
    print("chord: " + CATEGORIES[index])

    return CATEGORIES[index]


def getNotesFromChords(chord,signal,sr):
    notas_string = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    nota_2 =''
    nota_3 =''
   # octaveIndex = 2
    secondNoteIndex = 4
    octava = getOctaveFromChord(signal,sr)
    print("La octava es {}".format(octava))
    if '-' in chord :
      secondNoteIndex = 3
      if '#' in chord:
        chord = chord[0] + chord [1]
      else:
        chord = chord[0]
      
      	
    nota = chord[0]
   # octava = int(chord[octaveIndex])
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

    chord = chord + str(octava)
    triada = [chord, nota_2, nota_3]
     	 	
    return triada
    
def getOctaveFromChord(signal, sample_rate):
     # calculate FFT and absolute values

    X = fft(signal)
    X_mag = np.absolute(X)

    # generate x values (frequencies)
    f = np.linspace(0, sample_rate, len(X_mag))
    f_bins = int(len(X_mag)*0.1) 
    

    # find peaks in Y values. Use height value to filter lower peaks
    _, properties = find_peaks(X_mag, height=100)
    if len(properties['peak_heights']) > 0:
        y_peak = properties['peak_heights'][0]

        # get index for the peak
        peak_i = np.where(X_mag[:f_bins] == y_peak)

        # if we found an for a peak we print the note
        if len(peak_i) > 0:
            print("Peak es {}".format(f[peak_i[0]]))
            nota = str(librosa.hz_to_note(f[peak_i[0]]))
            nota = convertToNote(nota)
            octava = nota[1]
            if octava == "#":
                octava = nota[2]
            return int(octava)
    return 0

def convertToNote(val) :
    nota = str.replace(str.replace(val, "['", ""), "']", "")
    if len(nota) == 3 :
        nota = nota[0] + "#" + nota[2]

    return nota

if __name__ == "__main__":
    
    signal, sr = librosa.load(FILE_PATH)
    chord = getChordFromRNN(signal,sr)
    print(chord)
    print("Notes from Chord")
    print(getNotesFromChords(chord,signal,sr))
