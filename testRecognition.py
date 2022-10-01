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
import soundfile as sf

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
WORKSPACE = os.path.dirname(FILE_PATH)
MY_MODEL = keras.models.load_model('modelo-acordesv06.h5')
sys.path.insert(0, os.path.join(WORKSPACE, "input_parser"))



FILE_PATH = "Predict\Piano_B3_1658502790.1721418.wav"
ROOT_PATH = "Predict"
DATASET_PATH = "Data"
JSON_PATH = "data_chord.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 3 # measured in seconds
n_fft = 2048
hop_length = 512
SHAPE = 130

CATEGORIES = ["A","A#","A#-","A-","B","B-","C","C#", 
              "C#-","C-","D","D#","D#-","D-","E","E-",
              "F","F#","F#-","F-","G","G#","G#-","G-",
              "A","A#","A#-","A-","B","B-","C","C#", 
              "C#-","C-","D","D#","D#-","D-","E","E-",
              "F","F#","F#-","F-","G","G#","G#-","G-"]

INSTRUMENT = ["Guitar","Piano"]
NOTE_STRINGS = ["E2","A2","D3","G3","B3","E4"]
NOTE_CHROMATIC = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]


def correctShape(chroma_shape):
    return chroma_shape == 130

def normalizeShape(chroma_mat):
    nums = 0
    #init_shape tiene la dimension de columnas. 
    init_shape= chroma_mat.shape[1]
    #Me fijo cuantas columnas faltan por rellenar
    nums = SHAPE - init_shape
    #itero nums copiando el anterior
    arreglo = np.array(chroma_mat[:,init_shape-1])
   
    i = 0
    if nums > 0 :
        while i < nums :
            chroma_mat= np.column_stack((chroma_mat,arreglo))  
            i = i +1 
    else:
        chroma_mat = np.array(chroma_mat[:,: SHAPE])
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
    chord = "non_chord"
    index = 0 
    if len(signal) > 0 :
    # extract Chroma
        chroma = librosa.feature.chroma_cens(y=signal, sr=sample_rate,fmin=130,n_octaves=2)

        if not correctShape(chroma.shape[1]) :
            chroma = normalizeShape(chroma)
        if correctShape(chroma.shape[1]):
            chroma_reshape = tf.reshape(chroma, [ 1,12,130])
            my_prediction = MY_MODEL.predict(chroma_reshape)
            #print(my_prediction)
            index = np.argmax(my_prediction)
            chord = CATEGORIES[index]
    
            if index < 22 :
                instrument = INSTRUMENT[0]
            print("Instrument: {}\n".format( instrument))
            print("Chord: {}\n".format( chord))
            print("Octave: {}\n".format(getOctaveFromFundamental(signal,sample_rate)))
            if instrument == "Guitar":
                print(getNotesFromGuitar(signal,sample_rate,chord))
    
    return instrument, chord


def getNotesFromChords(chord,signal,sr):
    nota_2 =''
    nota_3 =''
   # octaveIndex = 2
    secondNoteIndex = 4
    #octava = getOctaveFromChord(signal,sr)
    octava = getOctaveFromFundamental(signal,sr)
    
    if '-' in chord :
      secondNoteIndex = 3
      if '#' in chord:
        chord = chord[0] + chord [1]
      else:
        chord = chord[0]
      
      	
    nota = chord[0]
   # octava = int(chord[octaveIndex])
    indice = NOTE_CHROMATIC.index(nota)
    if ((indice + secondNoteIndex) / 12) >= 1 :
        print(indice + secondNoteIndex)
        nota_2 = NOTE_CHROMATIC[((indice + secondNoteIndex )%12)] + str(octava + 1)
    else:
        nota_2 = NOTE_CHROMATIC[(indice + secondNoteIndex)]  + str(octava)
    if ((indice + 7) / 12) >= 1 :
        nota_3 = NOTE_CHROMATIC[((indice + 7) %12)] + str(octava + 1)
    else:
        nota_3 = NOTE_CHROMATIC[(indice + 7)]  + str(octava)

    chord = chord + str(octava)
    triada = [chord, nota_2, nota_3]
     	 	
    return triada
    

#Estimate octave from fundamental frecuency
def getOctaveFromFundamental(signal, sample_rate):
    octava = 4
    X = fft(signal)
    X_mag = np.absolute(X)

    # generate x values (frequencies)
    f = np.linspace(0, sample_rate, len(X_mag))
    f_bins = int(len(X_mag)*0.1) 
    

    # find peaks in Y values. Use height value to filter lower peaks
    _, properties = find_peaks(X_mag, height=100)
    if len(properties['peak_heights']) > 0:
        peak_sel = np.where(X_mag[:f_bins] == properties['peak_heights'][0])
        frec = f[peak_sel[0]]
    else:
        frec = 1001

    if frec < 130 :
        octava = 2
    elif frec >= 130 and frec < 250 :
        octava = 3
    elif frec >= 250 and frec < 530 :
        octava = 4
    elif frec >=523 and frec < 1000 :
        octava = 5
    print (frec)
    return octava

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
    ok_test_chord = 0
    ok_test_instrument = 0
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

    print("La cantidad de beats es: {}".format(len(beats)))
    onset_frames = librosa.onset.onset_detect(y, sr=sr, wait=10, pre_avg=10, post_avg=10, pre_max=30, post_max=30)
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
        if test_instrument == instrument : 
            ok_test_instrument = ok_test_instrument + 1
        if  test_chord == chord:
            ok_test_chord = ok_test_chord + 1
        audio_onsets = audio_onsets + 1
    ok_test = ok_test_chord
    accuracy = ((ok_test_chord  + ok_test_instrument)/2)/audio_onsets
    
    print("****RESUME****\n")
    print("Chord Selected: {}\n".format(chord))
    print("Instrument Selected: {}\n".format(instrument))
    print("Total Onsets: {}\n".format(audio_onsets))
    print("Total test chord success: {}\n".format(ok_test_chord))
    print("Total test instrument success: {}\n".format(ok_test_instrument))
    print("%\Accuracy: {}%\n".format(ok_test*100/audio_onsets))


def testSong(testing_path,notes,instrument):
    y, sr = librosa.load(testing_path)
    ok_test = 0
    audio_onsets = 0
    test_instrument = ""
    test_note = ""
    ok_test_note = 0
    ok_test_instrument = 0
    ONSET_PARAM = 20
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    print("La cantidad de beats es: {}".format(len(beats)))
    duration = librosa.get_duration(y=y, sr= sr)
  
    velocity_audio = 60*len(beats)/duration
    
    if velocity_audio > 40 and velocity_audio <= 80 :
        ONSET_PARAM = 10
    elif velocity_audio > 80 and velocity_audio <= 100:
        ONSET_PARAM = 7
    elif velocity_audio > 100:
        ONSET_PARAM = 5
    print("la velocidad del audio es: {}".format(velocity_audio))
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, max_size=5)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env,backtrack=True, normalize =True,wait=ONSET_PARAM, pre_avg=ONSET_PARAM, post_avg=ONSET_PARAM, pre_max=ONSET_PARAM, post_max=ONSET_PARAM)

    samples = librosa.frames_to_samples(onset_frames)

    # filter lower samples
    filteredSamples = filterLowSamples(samples)

    # get indexes of all samples in the numpy array
    indexes = np.where(filteredSamples>0)

    length = len(indexes[0])
    print("len samples {}".format(length))
    print("ONSET PARAMS: {}".format(ONSET_PARAM))
    j = 0
    onset_times = librosa.frames_to_time(onset_frames)
    print("Onset times: {}".format(onset_times))
    print("Len onset times: {}".format(len(onset_times)))
    print("Time difs:")

    for i in range(len(onset_times)):
        if i == len(onset_times)-1:
            continue
        print("X{} - X{} = {}".format(i+1,i,onset_times[i+1]-onset_times[i]))
    h = 0
    for i in indexes[0]:
        j = i
        if j < length-1:
            data = y[filteredSamples[j]:filteredSamples[j+1]]    
        elif j == length-1:
            data = y[filteredSamples[j]:]
        
        test_instrument, test_note = getChordandInstrumentFromRNN(data,sr)
        if test_instrument == instrument : 
            ok_test_instrument = ok_test_instrument + 1
        if  test_note == notes[h]:
            ok_test_note = ok_test_note + 1          
        sf.write("{}_{}_{}_{}".format(instrument,notes[h],time.time(),".wav"),data, sr)
        audio_onsets = audio_onsets + 1
        h = h + 1
    ok_test = ok_test_note
    accuracy = ((ok_test_note  + ok_test_instrument)/2)/audio_onsets
    
    print("****RESUME****\n")
    print("Note Selected: {}\n".format(note))
    print("Instrument Selected: {}\n".format(instrument))
    print("Total Onsets: {}\n".format(audio_onsets))
    print("Total test notes success: {}\n".format(ok_test_note))
    print("Total test instrument success: {}\n".format(ok_test_instrument))
    print("%\Accuracy: {}%\n".format(ok_test*100/audio_onsets))
    S = librosa.stft(y)
    logS = librosa.amplitude_to_db(abs(S))
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y, sr=sr)
    plt.vlines(onset_times, -0.8, 0.79, color='r', alpha=0.8) 

    plt.show()
    return
## Function that filters lower samples generated by input noise.
def filterLowSamples(samples):
    # find indexes of all elements lower than 2000 from samples
    indexes = np.where(samples < 2000)
    # remove elements for given indexes
    return np.delete(samples, indexes)

def getNotesFromGuitar(signal, sample_rate, chord):
    notes = []
         # calculate FFT and absolute values
    triad = getNotesFromChords(chord,signal,sample_rate)
    print(triad)
    note1 = cleanOctave(triad[0])
    note2 = cleanOctave(triad[1])
    note3 = cleanOctave(triad[2])
    triad = [note1, note2, note3]
    print(triad)
    X = fft(signal)
    X_mag = np.absolute(X)

    # generate x values (frequencies)
    f = np.linspace(0, sample_rate, len(X_mag))
    f_bins = int(len(X_mag)*0.1) 
    

    # find peaks in Y values. Use height value to filter lower peaks
    _, properties = find_peaks(X_mag, height=50)
    if len(properties['peak_heights']) > 0:
        y_peak = properties['peak_heights'][0]
        
        # get index for the peak
        peak_i = np.where(X_mag[:f_bins] == y_peak)
        #print("len: {}\n".format(len(properties['peak_heights'])))
        #print(properties['peak_heights'])
        # if we found an for a peak we print the note
        if len(peak_i) > 0:
            i = 0
            while  f[peak_i[0]] < 1100 and len(notes) < 6 and i < len(properties['peak_heights']):
                                
                nota = str(librosa.hz_to_note(f[peak_i[0]]))
                nota = convertToNote(nota)
                octava = nota [1]
                if octava == "#":
                    octava = nota [2]
                nota = cleanOctave(nota)
                nota_actual = nota
                
                #print(nota)
                if nota == triad[0]:
                    nota_actual = triad[0] + str(octava)
                    
                elif nota == triad[1]:
                    nota_actual = triad[1] + str(octava)
                    
                elif nota == triad[2]:
                    nota_actual = triad[2] + str(octava)

                if nota_actual is not nota :
                    try:
                        notes.index(nota_actual)
                    except:
                        notes.append(nota_actual)
                                    
                y_peak = properties['peak_heights'][i]
                i = i + 1
                # get index for the peak
                peak_i = np.where(X_mag[:f_bins] == y_peak)

    return notes

def setTablature(notes):
    tablature = [0,0,0,0,0,0]
    i = 0
    j = 0
    found = False
    position = 0

    while i < 6 :
        tablature[i] = getDistance(NOTE_STRINGS[i], notes[i])
        i = i + 1

    return tablature

def getDistance(note1, note2):
    octave1 = int(note1[1])
    if octave1 == "#":
        octave1 = int(note1[2])

    octave2 = int(note2[1])
    if octave2 == "#":
        octave2 = int(note2[2])

    note1 = cleanOctave(note1)
    note2 = cleanOctave(note2)
    distance = 0 
    index1 = NOTE_CHROMATIC.index(note1)
    index2 = NOTE_CHROMATIC.index(note2)
    if index1 > index2 :
        if octave1 - octave2 == 0:
            distance = 0
        else: 
            distance = (index2 + 12) - index1
    else:
        distance = index2 - index1
    return distance

def cleanOctave(note):
    nota = ""

    octava = note[1]
    if octava == "#":
        nota = note[0] + note[1]
    else:
        nota = note[0] 

    return nota

def getOctaveFromPeak(peak):
                
    nota = str(librosa.hz_to_note(peak))
    nota = convertToNote(nota)
    octava = nota[1]
    if octava == "#":
        octava = nota[2]
    return int(octava)
   

if __name__ == "__main__":
    
    print("***WELCOME***")
    noteIndex = input("\nSelect a note:\n[0] - C\n[1] - D\n[2] - E\n[3] - F\n[4] - G\n[5] - A\n[6] - B\n>")

    note = ""
    match noteIndex:
        case "0":
            note = "C"
        case "1": 
            note = "D"
        case "2":
            note = "E"
        case "3":
            note = "F"
        case "4":
            note = "G"
        case "5":
            note = "A"
        case "6":
            note = "B"            
        case _:
            print("Please don't be retarded")
            quit()
    sustainIndex = input("\sustain:\n[1] - YES\n[2] - NO\n[->]:>")

    match sustainIndex:
        case "1":
            note = note + '#'
        case "2": 
            note = note

    minorIndex = input("\nminor:\n[1] - YES\n[2] - NO\n[->]:>")

    
    match minorIndex:
        case "1":
            note = note + "-"
        case "2": 
            note = note
   
    instrumentIndex = input("Instrument:\n[1] - GUITAR\n[2] - PIANO\n[->]:>")
   # instrument = "Guitar"
   
    match instrumentIndex:
        case "1":
            instrument = "Guitar"
        case "2": 
            instrument = "Piano"

    testIndex = input("\nSelect a test:\n[0] - Batch\n[1] - Only One\n[2] - Onset\n[3] - Test song\n [4] - Exit:\n>:")
    match testIndex:
        case "0":
            checkBach(FILE_PATH,note,instrument)    
        case "1": 
            signal, sr = librosa.load(FILE_PATH)
            instrumento, chord = getChordandInstrumentFromRNN(signal,sr)
            notes = []
            print("TESTING PREDICTION COME HERE\n")
            print("The instrument is {}\n".format(instrument))
            print("The chord is {}\n".format(chord))
            print("The octave is {}\n".format(getOctaveFromFundamental(signal,sr)))
            print("And the notes from Chord\n")
            if instrument == INSTRUMENT[1]:
                print(getNotesFromChords(chord,signal,sr))
            else:
                notes = getNotesFromGuitar(signal,sr,chord)
                print(notes)
                #print("FINALLY, THE TABLATURE :o\n")
                print(setTablature(notes))
            print("PLEASE FOLLOW WITH THE TEST! \n")

        case "2":
            largeAudioWithOnset(FILE_PATH,note,instrument)
        case "3":
            note_song =["G","G","D","D","D","D","G","G","C","C",
                        "G","G","D","D","G"]
            testSong(FILE_PATH,note_song,instrument)
            print("See you\n")
            quit()
        case "4":
            print("See you\n")
            quit()

