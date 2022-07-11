from array import ArrayType
import keras
import librosa
import numpy as np
import math
import os
import json
import tensorflow
from sklearn.model_selection import train_test_split

FILE_PATH = "Predict/B_maj_2_0.wav"
DATASET_PATH = "Data"
JSON_PATH = "data_instruments.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 3 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
num_mfcc = 13
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
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data succesfully loaded!")

    return X, y

def load_data(data_path):
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data succesfully loaded!")

    return X, y
    
def open_mfcc(FILE_PATH, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """Open MFCCs from music dataset and saves them into a json file along witgh genre labels.
        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        :return:
        """
  
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
    print(num_mfcc_vectors_per_segment)
    # loop through all genre sub-folder
   

    
    signal, sample_rate = librosa.load(FILE_PATH, sr=SAMPLE_RATE, duration=3)

     # process all segments of audio file
    for d in range(num_segments):

                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # extract mfcc
                    mfcc = librosa.feature.mfcc(y =signal[start:finish], 
                                                sr = SAMPLE_RATE, 
                                                n_mfcc=num_mfcc, 
                                                n_fft=n_fft, 
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    # store only mfcc feature with expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                            array_mfcc = np.array(mfcc.tolist())
                            print("{}, segment:{}".format(FILE_PATH, d+1))
                    
    tensorflow.expand_dims(array_mfcc, [0])
    return array_mfcc


        


my_model = keras.models.load_model('modelo-instrumentos.h5')
#my_value_test_X, my_value_test_Y = load_data("data_piano.json")
#X_train, X_test, y_train, y_test = train_test_split(my_value_test_X, my_value_test_Y, test_size=0.3)
signal, sr = librosa.load(FILE_PATH, sr=SAMPLE_RATE)
mel_spec = librosa.feature.melspectrogram(y =signal, 
                                                sr = SAMPLE_RATE,                                                 
                                                n_fft=n_fft, 
                                                hop_length=hop_length,
                                                n_mels=15, fmax=4000)
                #mel_spec = mel_spec.T
sibebe = librosa.power_to_db(mel_spec)
my_prediction = my_model.predict(0,sibebe[1],sibebe[2])
print(len(my_prediction))