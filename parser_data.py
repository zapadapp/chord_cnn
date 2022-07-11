import json
import os
import math
from tkinter.ttk import LabeledScale
from attr import s
import librosa

DATASET_PATH = "Data"
JSON_PATH = "data_chord.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 3 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_chroma(dataset_path, json_path, n_fft=2048, hop_length=512):
    """Extracts Chroma from music dataset and saves them into a json file along witgh Instruments labels.
        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save CHROMA
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :return:
        """

    # dictionary to store mapping, labels, and Mel spectrogram
    data = {
        "mapping": [],
        "labels": [],
        "chroma": []
    }

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a chord sub-folder level
        if dirpath is not dataset_path and isWavDir(dirpath):

            # save chord label (i.e., sub-folder name) in the mapping
            label = generateLabel(dirpath)
            data["mapping"].append(label)
            print("\nProcessing: {}".format(label))

            # process all audio files in chord sub-dir
            for f in filenames:

		# load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
                record = os.path.split(file_path)[1]
                # extract Chroma
                chroma = librosa.feature.chroma_cens(y=signal, sr=sample_rate)
            
                data["chroma"].append(chroma.tolist())
                data["labels"].append(i-1)
                print("{}, file:{} shape:{}".format(label, record,chroma.shape))

    # save CHROMA to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        
def isWavDir(datapath):
    return len(os.path.join(datapath, os.path.split(datapath)[1], '*.wav')) > 0

def generateLabel(datapath):
     dir_labels = datapath.split("\\")
     size_dir = len(dir_labels)

     semantic_label = dir_labels[size_dir - 2] + "-"+ dir_labels[size_dir - 1]
     return semantic_label

if __name__ == "__main__":
    save_chroma(DATASET_PATH, JSON_PATH)
