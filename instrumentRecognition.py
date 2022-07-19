from array import ArrayType
import keras
import librosa
import librosa.display 
import numpy as np
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt


FILE_PATH = "Data/Piano/Mayor/LA/Piano_A4_1657928571.3672063.wav"
DATASET_PATH = "Data"
JSON_PATH = "data_chord.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 3 # measured in seconds
n_fft = 2048
hop_length = 512
CATEGORIES = [
    "Piano-Dim",
    "Mayor-DO",
    "Mayor-DOs",
    "Mayor-FA",
    "Mayor-LA",
    "Mayor-MI",
    "Mayor-MIb",
    "Mayor-RE",
    "Mayor-SI",
    "Mayor-SIb",
    "Mayor-SOL",
    "Mayor-SOLs",
    "Menor-DO",
    "Menor-DOs",
    "Menor-FA",
    "Menor-FAs",
    "Menor-LA",
    "Menor-MI",
    "Menor-MIb",
    "Menor-RE",
    "Menor-SI",
    "Menor-SIb",
    "Menor-SOL",
    "Menor-SOLs"
]

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

    
my_model = keras.models.load_model('modelo-acordes-v01.h5')

signal, sample_rate = librosa.load(FILE_PATH, sr=SAMPLE_RATE)

# extract Chroma
chroma = librosa.feature.chroma_cens(y=signal, sr=sample_rate)
if not correctShape(chroma.shape[1]) :
   chroma = normalizeShape(chroma)
   
chroma_reshape = tf.reshape(chroma, [ 1,12,130])
my_prediction = my_model.predict(chroma_reshape)
print(my_prediction)
index = np.argmax(my_prediction)
print("chord: " + CATEGORIES[index])

#plt.figure(figsize=(25, 10))
#librosa.display.specshow(chroma, 
#                         y_axis="chroma", 
#                         x_axis="time",
#                         sr=sample_rate)
#plt.colorbar(format="%+2.f")
#plt.show()