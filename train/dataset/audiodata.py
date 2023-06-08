import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

"""
Output : Randomly cropped wave with specific length & corresponding f0 (if necessary).
"""


class AudioData(Dataset):
    def __init__(
        self,
        paths,
        seed=940513,
        waveform_sec=4.0,
        sample_rate=16000,
        waveform_transform=None,
        label_transform=None,
        max_voices=10
    ):
        super().__init__()
        self.paths = paths
        self.random = np.random.RandomState(seed)
        self.waveform_sec = waveform_sec
        self.waveform_transform = waveform_transform
        self.label_transform = label_transform
        self.sample_rate = sample_rate
        self.max_voices = max_voices

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        return len(self.paths)


class SupervisedAudioData(AudioData):
    def __init__(
        self,
        paths,
        csv_paths,
        seed=940513,
        waveform_sec=1.0,
        sample_rate=16000,
        frame_resolution=0.004,
        waveform_transform=None,
        label_transform=None,
        random_sample=True,
        max_voices=10
    ):
        super().__init__(
            paths=paths,
            seed=seed,
            waveform_sec=waveform_sec,
            sample_rate=sample_rate,
            waveform_transform=waveform_transform,
            label_transform=label_transform,
            max_voices=max_voices,
        )
        self.csv_paths = csv_paths
        self.frame_resolution = frame_resolution
        self.num_frame = int(self.waveform_sec / self.frame_resolution)  # number of csv's row
        self.hop_length = int(self.sample_rate * frame_resolution)
        self.num_wave = int(self.sample_rate * self.waveform_sec)
        self.random_sample = random_sample
        self.max_voices = max_voices

    def __getitem__(self, file_idx):
        target_f0 = pd.read_csv(self.csv_paths[file_idx])
        target_f0['frequency'] = target_f0['frequency'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))
        target_f0['velocity'] = target_f0['velocity'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))

        # sample interval
        if self.random_sample:
            idx_from = self.random.randint(
                1, len(target_f0) - self.num_frame
            )  # No samples from first frame - annoying to implement b.c it has to be padding at the first frame.
        else:
            idx_from = 1
        idx_to = idx_from + self.num_frame
        frame_from = target_f0["time"][idx_from]
        # frame_to = target_f0['time'][idx_to]

        voices = np.vstack(target_f0['frequency']).shape[1]

        f0 = np.zeros((self.num_frame, self.max_voices), dtype=np.float32)
        f0[:, :voices] = np.vstack(target_f0["frequency"][idx_from:idx_to].values).astype(np.float32)
        f0 = torch.from_numpy(f0).transpose(-2,-1)

        velocity = np.zeros((self.num_frame, self.max_voices), dtype=np.float32)
        velocity[:, :voices] = np.vstack(target_f0["velocity"][idx_from:idx_to].values).astype(np.float32)
        velocity = torch.from_numpy(velocity).transpose(-2,-1)

        waveform_from = int(frame_from * self.sample_rate)
        # waveform_to = waveform_from + self.num_wave

        audio, sr = torchaudio.load(
            self.paths[file_idx], frame_offset=waveform_from, num_frames=self.num_wave
        )

        audio = audio[0]
        assert sr == self.sample_rate

        return dict(audio=audio, f0=f0, velocity=velocity)
