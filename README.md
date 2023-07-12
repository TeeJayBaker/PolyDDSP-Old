# PolyDDSP : Polyphonic generalisation of the DDSP model

based on the implementation provided by sweetcocoa
- Jongho Choi (sweetcocoa@snu.ac.kr, BS Student @ Seoul National Univ.)
- Sungho Lee (dlfqhsdugod1106@gmail.com, BS Student @ Postech.)

# DDSP : Differentiable Digital Signal Processing

> Original Authors : Jesse Engel, Lamtharn (Hanoi) Hantrakul, Chenjie Gu, Adam Roberts (Google)

## Demo Page of sweetcocoa's implementation ##

- [Link](https://sweetcocoa.github.io/ddsp-pytorch-samples/)

## How to train with your own data

1. Clone this repository

```bash
git clone https://github.com/TeeJayBaker/PolyDDSP
```

2. Prepare your own audio data. (wav, mp3, flac.. )
3. Use ffmpeg to convert that audio's sampling rate to 16k

```bash
# example
ffmpeg -y -loglevel fatal -i $input_file -ac 1 -ar 16000 $output_file
```
4. Use [CREPE](https://github.com/marl/crepe) to precalculate the fundamental frequency of the audio.

```bash
# example
crepe directory-to-audio/ --output directory-to-audio/f0_0.004/  --viterbi --step-size 4
```

5. MAKE config file. (See configuration *config/violin.yaml* to make appropriate config file.) And edit train/train.py

```python
config = setup(default_config="../configs/your_config.yaml")
```
6. Run train/train.py

```bash
cd train
python train.py
```

## How to test your own model ##

```bash
cd train
python test.py\ 
--input input.wav\
--output output.wav\
--ckpt trained_weight.pth\
--config config/your-config.yaml\
--wave_length 16000
```


