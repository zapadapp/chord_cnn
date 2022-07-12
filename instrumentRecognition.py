from array import ArrayType
import keras
import librosa
import librosa.display 
import numpy as np
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt


FILE_PATH = "Data/Piano/Mayor/DO/C_maj_2_0.wav"
DATASET_PATH = "Data"
JSON_PATH = "data_chord.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 3 # measured in seconds
n_fft = 2048
hop_length = 512

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
chroma_reshape = tf.reshape(chroma, [ 1,12,130])
my_prediction = my_model.predict(chroma_reshape)
print(my_prediction)
#plt.figure(figsize=(25, 10))
#librosa.display.specshow(chroma, 
#                         y_axis="chroma", 
#                         x_axis="time",
#                         sr=sample_rate)
#plt.colorbar(format="%+2.f")
#plt.show()