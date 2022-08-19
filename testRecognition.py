from array import ArrayType
from itertools import count
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
import time 
import pyaudio
import webrtcvad
import pyaudio
import wave
import os
from warnings import simplefilter
import time



FILE_PATH = os.path.dirname(os.path.realpath(__file__))
WORKSPACE = os.path.dirname(FILE_PATH)
MY_MODEL = keras.models.load_model('modelo-acordes-v04.h5')
sys.path.insert(0, os.path.join(WORKSPACE, "input_parser"))



FILE_PATH = "Predict\Guitar_E4_1660870899.388045.wav"
ROOT_PATH = "Predict"
DATASET_PATH = "Data"
JSON_PATH = "data_chord.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 3 # measured in seconds
n_fft = 2048
hop_length = 512

CATEGORIES = ["A","A-","B","B-","C", "C-","D", "D-","E","E-",
              "F","F-","G","G-",
              "A","A#","A#-","A-","B","B-","C","C#","C#-","C-",
              "D","D#","D#-","D-","E","E-","F","F#","F#-","F-",
              "G","G#","G#-","G-"]

INSTRUMENT = ["Guitar","Piano"]

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

def checkBach(testing_path, chord, instrument):
    test_instrument = "init"
    test_chord = "init"
    count = 0
    success = 0
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(testing_path)):
     
        for f in filenames:
		# load audio file
            file_path = os.path.join(dirpath, f)
            signal, sr = librosa.load(file_path)
            test_instrument, test_chord = getChordandInstrumentFromRNN(signal,sr)
            if test_instrument == instrument and test_chord == chord:
                success = success +1
            count = count + 1 
    print("****RESUME****\n")
    print("Chord Selected: {}\n".format(chord))
    print("Instrument Selected: {}\n".format(instrument))
    print("Total in Bach: {}\n".format(count))
    print("Total test success: {}\n".format(success))
    print("%\Accuracy: {}%\n".format(success*100/count))
    return

def getChordandInstrumentFromRNN(signal, sample_rate):
    
    instrument = INSTRUMENT[1]

    # extract Chroma
    chroma = librosa.feature.chroma_cens(y=signal, sr=sample_rate,fmin=130,n_octaves=2)

    if not correctShape(chroma.shape[1]) :
        chroma = normalizeShape(chroma)

    chroma_reshape = tf.reshape(chroma, [ 1,12,130])
    my_prediction = MY_MODEL.predict(chroma_reshape)
    #print(my_prediction)
    index = np.argmax(my_prediction)
    chord = CATEGORIES[index]
    
    if index < 14 :
        instrument = INSTRUMENT[0]
    print("Instrument {}\n".format( instrument))
    print("Chord {}\n".format( chord))
    
    return instrument, chord


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

def largeAudioWithOnset(FILE_PATH,chord,instrument):
      
    y, sr = librosa.load(FILE_PATH)
    ok_test = 0
    audio_onsets = 0
    test_instrument = ""
    test_chord = ""

    onset_frames = librosa.onset.onset_detect(y, sr=sr, wait=1, pre_avg=1, post_avg=1, pre_max=1, post_max=1)
    samples = librosa.frames_to_samples(onset_frames)

    # filter lower samples
    filteredSamples = filterLowSamples(samples)

    # get indexes of all samples in the numpy array
    indexes = np.where(filteredSamples>0)

    length = len(indexes[0])
    print("len samples {}".format(length))
    j = 0
 
    for i in indexes[0]:
        j = i
        if j < length-1:
            test_instrument, test_chord = getChordandInstrumentFromRNN(y[filteredSamples[j]:filteredSamples[j+1]],sr)    
        elif j == length-1:
            test_instrument, test_chord = getChordandInstrumentFromRNN(y[filteredSamples[j]:],sr)
        if test_instrument == instrument and test_chord == chord : 
           ok_test = ok_test + 1
        audio_onsets = audio_onsets + 1
    print("****RESUME****\n")
    print("Chord Selected: {}\n".format(chord))
    print("Instrument Selected: {}\n".format(instrument))
    print("Total Onsets: {}\n".format(audio_onsets))
    print("Total test success: {}\n".format(ok_test))
    print("%\Accuracy: {}%\n".format(ok_test*100/audio_onsets))



## Function that filters lower samples generated by input noise.
def filterLowSamples(samples):
    # find indexes of all elements lower than 2000 from samples
    indexes = np.where(samples < 2000)
    # remove elements for given indexes
    return np.delete(samples, indexes)


    
if __name__ == "__main__":
    
    print("***WELCOME***")
    instrumentIndex = input("\nSelect a test:\n[0] - Batch\n[1] - Only One\n[2] - Onset\n[3] - Exit:\n>:")
    match instrumentIndex:
        case "0":
            checkBach(FILE_PATH,"A","Guitar")    
        case "1": 
            signal, sr = librosa.load(FILE_PATH)
            instrument, chord = getChordandInstrumentFromRNN(signal,sr)
            print("TESTING PREDICTION COME HERE\n")
            print("The instrument is {}\n".format(instrument))
            print("The chord is {}\n".format(chord))
            print("And the notes from Chord\n")
            print(getNotesFromChords(chord,signal,sr))
            print("PLEASE FOLLOW WITH THE TEST! \n")
        case "2":
            largeAudioWithOnset(FILE_PATH,"E","Guitar")
        case "3":
            print("See you\n")
            quit()

