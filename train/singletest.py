import torch
import torch.nn as nn
import torchaudio
from omegaconf import OmegaConf
import sys, os, tqdm, glob
import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim

from trainer.trainer import Trainer
from trainer.io import setup, set_seeds

from dataset.audiodata import SupervisedAudioData, AudioData
from network.autoencoder.autoencoder import AutoEncoder
from loss.mss_loss import MSSLoss
from optimizer.radam import RAdam

"""
"setup" allows you to OVERRIDE the config through command line interface
- for example
$ python train.py --batch_size 64 --lr 0.01 --use_reverb
"""

ckpt = "../../ckpt/violin/soloviolin.pth-100000"
config = setup(default_config="../configs/guitar.yaml")
# config = setup(pdb_on_error=True, trace=False, autolog=False, default_config=dict(
#     # general config
#     ckpt="../../ddsp_ckpt/violin/200131.pth",  # checkpoint
#     gpu="0",
#     num_workers=4,  # number of dataloader thread
#     seed=940513,    # random seed
#     tensorboard_dir="../tensorboard_log/",
#     experiment_name="DDSP_violin",   # experiment results are compared w/ this name.

#     # data config
#     train="../data/violin/train/",  # data directory. should contain f0, too.
#     test="../data/violin/test/",
#     waveform_sec=1.0,   # the length of training data.
#     frame_resolution=0.004,   # 1 / frame rate
#     batch_size=64,
#     f0_threshold=0.5,    # f0 with confidence below threshold will go to ZERO.
#     valid_waveform_sec=4.0,  # the length of validation data
#     n_fft=2048,    # (Z encoder)
#     n_mels=128,    # (Z encoder)
#     n_mfcc=30,     # (Z encoder)
#     sample_rate=16000,

#     # training config
#     num_step=100000,
#     validation_interval=1000,
#     lr=0.001,
#     lr_decay=0.98,
#     lr_min=1e-7,
#     lr_scheduler="multi", # 'plateau' 'no' 'cosine'
#     optimizer='radam',   # 'adam', 'radam'
#     loss="mss",
#     metric="mss",
#     resume=False,    # when training from a specific checkpoint.

#     # network config
#     mlp_units=512,
#     mlp_layers=3,
#     use_z=False,
#     use_reverb=False,
#     z_units=16,
#     n_harmonics=101,
#     n_freq=65,
#     gru_units=512,
#     crepe="full",
#     bidirectional=False,
#     ))

print(OmegaConf.create(config.__dict__))
set_seeds(config.seed)
Trainer.set_experiment_name(config.experiment_name)

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 
    "cpu")

net = AutoEncoder(config).to(device)
net.load_state_dict(torch.load(ckpt))
net.eval()

loss = MSSLoss([2048, 1024, 512, 256], use_reverb=config.use_reverb).to(device)

# Define evaluation metrics
if config.metric == "mss":

    def metric(output, gt):
        with torch.no_grad():
            return -loss(output, gt)


elif config.metric == "f0":
    # TODO Implement
    raise NotImplementedError
else:
    raise NotImplementedError
# -----------------------------/>

# Dataset & Dataloader Prepare
train_data = glob.glob(config.train + "/*.wav") * config.batch_size
train_data_csv = [
    os.path.dirname(wav)
    + f"/f0_{config.frame_resolution:.3f}/"
    + os.path.basename(os.path.splitext(wav)[0])
    + ".f0.csv"
    for wav in train_data
]

valid_data = glob.glob(config.test + "/*.wav")
valid_data_csv = [
    os.path.dirname(wav)
    + f"/f0_{config.frame_resolution:.3f}/"
    + os.path.basename(os.path.splitext(wav)[0])
    + ".f0.csv"
    for wav in valid_data
]

train_dataset = SupervisedAudioData(
    sample_rate=config.sample_rate,
    paths=train_data,
    csv_paths=train_data_csv,
    seed=config.seed,
    waveform_sec=config.waveform_sec,
    frame_resolution=config.frame_resolution,
    max_voices=10
)

valid_dataset = SupervisedAudioData(
    sample_rate=config.sample_rate,
    paths=valid_data,
    csv_paths=valid_data_csv,
    seed=config.seed,
    waveform_sec=config.valid_waveform_sec,
    frame_resolution=config.frame_resolution,
    random_sample=False,
    max_voices=10
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
    pin_memory=True,
)

valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=int(config.batch_size // (config.valid_waveform_sec / config.waveform_sec)),
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=False,
)
# -------------------------------------/>

# Setting Optimizer
if config.optimizer == "adam":
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=config.lr)
elif config.optimizer == "radam":
    optimizer = RAdam(filter(lambda x: x.requires_grad, net.parameters()), lr=config.lr)
else:
    raise NotImplementedError
# -------------------------------------/>

# Setting Scheduler
if config.lr_scheduler == "cosine":
    # restart every T_0 * validation_interval steps
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, eta_min=config.lr_min
    )
elif config.lr_scheduler == "plateau":
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=config.lr_decay
    )
elif config.lr_scheduler == "multi":
    # decay every ( 10000 // validation_interval ) steps
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        [(x + 1) * 10000 // config.validation_interval for x in range(10)],
        gamma=config.lr_decay,
    )
elif config.lr_scheduler == "no":
    scheduler = None
else:
    raise ValueError(f"unknown lr_scheduler :: {config.lr_scheduler}")
# ---------------------------------------/>
import pandas as pd

test = train_dataset.__getitem__(1)

torchaudio.save('input.wav', test['audio'].unsqueeze(0), 16000)

test['audio'] = test['audio'].unsqueeze(0).to(device)
test['f0'] = test['f0'].unsqueeze(0).to(device)
test['velocity'] = test['velocity'].unsqueeze(0).to(device)

torch.set_printoptions(profile='full')
print(test['f0'])

decoder = net.decoder
encoder = net.encoder

# # print(test["f0"].shape)

test2 = encoder(test)

test3 = decoder(test2)

# # print(test3['H'].shape)

harmonic_oscillator = net.harmonic_oscillator
filtered_noise = net.filtered_noise
reverb = net.reverb

harmonic = harmonic_oscillator(test3)
noise = filtered_noise(test3)

audio = dict(harmonic=harmonic, noise=noise, audio_synth=harmonic + noise[:, : harmonic.shape[-1]])
audio["audio_reverb"] = reverb(audio)

# print(test4)

torchaudio.save('harmonic.wav', audio["harmonic"].cpu(), 16000)
torchaudio.save('noise.wav', audio["noise"].cpu(), 16000)
torchaudio.save('audio_synth.wav', audio["audio_synth"].cpu(), 16000)
torchaudio.save('audio_reverb.wav', audio["audio_reverb"].cpu(), 16000)

# filtered_noise = net.filtered_noise

# test5 = filtered_noise(test3)

# print(test5)

# test6 = net(test)

# print(test6)
