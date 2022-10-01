import json
import os
import sys 
from tkinter.ttk import LabeledScale
import librosa
from matplotlib.pyplot import axis
import numpy as np
import time

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
WORKSPACE = os.path.dirname(FILE_PATH)

sys.path.insert(0, os.path.join(WORKSPACE, "instrumentsDataset"))


DATASET_PATH = "C:/Users/Juanma/Desktop/ZapadAPP/Workspace/instrumentsDatasets/ChordTrain"
JSON_PATH = "data_chord.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 3 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
SHAPE = 130


def isWavDir(datapath):
    list_dir = os.listdir(datapath)
    for x in list_dir:
        if '.wav' in x :
            return True
    return False

def generateLabel(datapath):
     dir_labels = datapath.split("\\")
     size_dir = len(dir_labels)

    # semantic_label = dir_labels[size_dir - 2] + "-"+ dir_labels[size_dir - 1]
     semantic_label = dir_labels[size_dir -2 ] + "-" + dir_labels[size_dir - 1]
    
     return semantic_label

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
        

def save_chroma(dataset_path, json_path):

    data = {
        "mapping": [],
        "labels": [],
        "chroma": []
    }
    count_label = 0 
    total_files = 0
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a chord sub-folder level
        if dirpath is not dataset_path and isWavDir(dirpath):
            #genero el label.
            label = generateLabel(dirpath)
            data["mapping"].append(label)

            print("\nProcessing: {}\nLabel Assign: {}\n".format(label, count_label))

            # process all audio files in chord sub-dir
            for f in filenames:

		        # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
                duration = librosa.get_duration(y=signal, sr= sample_rate)
                record = os.path.split(file_path)[1]
                #onSet Detect Here
                if duration > 3.0:
                    ONSET_PARAM = 20
                    tempo, beats = librosa.beat.beat_track(y=signal, sr=SAMPLE_RATE)
                   
                    velocity_audio = 60*len(beats)/duration

                    if velocity_audio > 40 and velocity_audio <= 80 :
                        ONSET_PARAM = 10
                    elif velocity_audio > 80 and velocity_audio <= 100:
                        ONSET_PARAM = 7
                    elif velocity_audio > 100:
                        ONSET_PARAM = 5
                    onset_frames = librosa.onset.onset_detect(y=signal, sr=sample_rate, normalize= True, wait=ONSET_PARAM, pre_avg=ONSET_PARAM, post_avg=ONSET_PARAM, pre_max=ONSET_PARAM, post_max=ONSET_PARAM)

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
                            onset_signal = signal[filteredSamples[j]:filteredSamples[j+1]]
                        elif j == length-1:
                            onset_signal = signal[filteredSamples[j]:]

                        chroma = librosa.feature.chroma_cens(y=onset_signal, sr=sample_rate,fmin=130,n_octaves=2)

                        if not correctShape(chroma.shape[1]) : 
                            chroma =  normalizeShape(chroma)

                        data["chroma"].append(chroma.tolist())
                        data["labels"].append(count_label)
                        print("{}, file:{} shape:{}".format(label, record,chroma.shape))
                        total_files = total_files + 1
                else:
                # extract Chroma
                    chroma = librosa.feature.chroma_cens(y=signal, sr=sample_rate,fmin=130,n_octaves=2)

                    if not correctShape(chroma.shape[1]) : 
                       chroma =  normalizeShape(chroma)

                    data["chroma"].append(chroma.tolist())
                    data["labels"].append(count_label)
                    print("{}, file:{} shape:{}".format(label, record,chroma.shape))
                    total_files = total_files + 1 
            count_label = count_label + 1
                # save CHROMA to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
    return total_files


## Function that filters lower samples generated by input noise.
def filterLowSamples(samples):
    # find indexes of all elements lower than 2000 from samples
    indexes = np.where(samples < 2000)
    # remove elements for given indexes
    return np.delete(samples, indexes)


if __name__ == "__main__":
    init_time = time.time()
    total_files = save_chroma(DATASET_PATH, JSON_PATH)
    finish_time = time.time()
    total_time = finish_time - init_time
    seconds = int(total_time %60)
    minutes = int((total_time/60)%60)
    hours = int(total_time /3600)
    print("RESUME\n")
    print("Total files processing: {}\n".format(total_files))
    print("Time processing: {}:{}:{}".format(hours,minutes,seconds))


 