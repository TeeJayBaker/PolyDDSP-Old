import pandas as pd

test = pd.read_csv("00_BN1-147-Gb_solo_mic_basic_pitch.csv", usecols=[0,1,2,3], header=0, index_col=False).sort_values('start_time_s', ignore_index=True)
print(test)

import torch
import torch.nn as nn
import numpy as np
import torchaudio
from tqdm import tqdm

audio, sr = torchaudio.load(
            "00_BN1-147-Gb_solo_mic.wav"
        )

audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=16000)

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

print(overlap_fix(test))
          
def change_notation_format(audio, csv):
    
    num_frames = int((audio.size(dim=1)/16000)/0.004)

    time = np.linspace(0, (num_frames-1) * 0.004, num_frames)

    voices, num_voice = overlap_fix(csv)

    new = {'time': time, 
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

test2 = change_notation_format(audio, test)





fundamentals = torch.tensor(test2['frequency']).unsqueeze(0)
velocity = torch.tensor(test2['velocity']).unsqueeze(0)

num_osc = 20

partials = torch.linspace(1, num_osc, num_osc, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)

f0_bank = fundamentals.unsqueeze(-1).expand(-1, -1, -1, num_osc).transpose(1, 3) * partials

temp = f0_bank.reshape(f0_bank.size(0), f0_bank.size(1)*f0_bank.size(2), f0_bank.size(3))

print(temp[:,:,200])

envelope = torch.linspace(1, num_osc, num_osc, dtype=torch.float32).unsqueeze(-1).unsqueeze(0).expand(-1,-1,len(test2['frequency']))

print(envelope.repeat_interleave(4,dim=1)[:,:,0])

framerate_to_audiorate = nn.Upsample(
            scale_factor=64, mode="linear", align_corners=False
        )

def poly_upsample(tensor):
    #unstack polyphony
    tensors = list(tensor.unbind(2))

    #upsample each note separately
    for i in range(len(tensors)):
        tensors[i] = framerate_to_audiorate(tensors[i])
    
    #recombine
    tensor = torch.stack(tensors, dim=2)
    return tensor

def poly_upsample2(tensor):
    a,b,c,d = tensor.size(0), tensor.size(1), tensor.size(2), tensor.size(3)
    tensor = tensor.reshape(a, b*c, d)
    tensor = framerate_to_audiorate(tensor)
    tensor = tensor.reshape(a,b,c,tensor.size(2))
    return tensor


f0_bank = poly_upsample2(f0_bank)
