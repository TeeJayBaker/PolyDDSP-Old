import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
import os, glob, torchaudio

data = glob.glob("data/guitar/train/*.wav")
data_csv = [
    os.path.dirname(wav)
    + "/f0_0.004/"
    + os.path.basename(os.path.splitext(wav)[0])
    + "_basic_pitch.csv"
    for wav in data
]

def overlap_fix(csv):
    #takes notes df and outputs list of none overlapping voices index
    voices = []
    num_voice = 0
    to_sort = list(csv.index.values)

    while to_sort != []:
        end = 0.0
        voices.append([])
        num_voice += 1
        unsorted = []
        for index in to_sort:
            if csv.iloc[index]['start_time_s'] >= end:
                voices[num_voice - 1].append(index)
                end = csv.iloc[index]['end_time_s']
            else:
                unsorted.append(index)

        to_sort = unsorted

    return voices, num_voice

          
def change_notation_format(audio, csv):

    num_frames = int((audio.size(dim=1)/16000)/0.004)
    print(audio.size(dim=1))
    time = np.linspace(0, (num_frames-1) * 0.004, num_frames, )

    voices, num_voice = overlap_fix(csv)

    new = {'time': np.around(time,3), 
           'frequency': np.zeros((num_frames, num_voice)), 
           'velocity': np.zeros((num_frames, num_voice))}

    for i in range(num_voice):
        for dfindex in voices[i]:
            start = int(np.ceil(csv.iloc[dfindex]['start_time_s']/0.004))
            end = int(np.ceil(csv.iloc[dfindex]['end_time_s']/0.004))
            for index in range(start,end):
                new['frequency'][index][i] = 440 * (2 ** ((csv.iloc[dfindex]['pitch_midi']-69)/12))
                new['velocity'][index][i] = csv.iloc[dfindex]['velocity']/127                             

    return new

from io import StringIO

with tqdm(range(len(data_csv))) as t:
    for i in t:
        t.set_description(data[i])
        audio, _ = torchaudio.load(data[i]) 
        with open(data_csv[i], 'r') as f:
            lines = [line[:60] for line in f.readlines()] # truncate each line to 50 characters
            truncated_csv = '\n'.join(lines) # join the truncated lines into a string
        csv = pd.read_csv(StringIO(truncated_csv), usecols=[0,1,2,3], header=0, index_col=False, engine='c').sort_values('start_time_s', ignore_index=True)
        csv = change_notation_format(audio, csv)
        df = pd.DataFrame.from_dict(csv, orient='index').transpose()
        df.to_csv(data_csv[i][:-16] + '.f0.csv', index=False)
