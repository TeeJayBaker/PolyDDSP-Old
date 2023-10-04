"""
Implementation of decoder network architecture of DDSP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    MLP (Multi-layer Perception). 

    One layer consists of what as below:
        - 1 Dense Layer
        - 1 Layer Norm
        - 1 ReLU

    constructor arguments :
        n_input : dimension of input
        n_units : dimension of hidden unit
        n_layer : depth of MLP (the number of layers)
        relu : relu (default : nn.ReLU, can be changed to nn.LeakyReLU, nn.PReLU for example.)

    input(x): torch.tensor w/ shape(B, ... , n_input)
    output(x): torch.tensor w/ (B, ..., n_units)
    """

    def __init__(self, n_input, n_units, n_layer, relu=nn.ReLU, inplace=True):
        super().__init__()
        self.n_layer = n_layer
        self.n_input = n_input
        self.n_units = n_units
        self.inplace = inplace

        self.add_module(
            f"mlp_layer1",
            nn.Sequential(
                nn.Linear(n_input, n_units),
                nn.LayerNorm(normalized_shape=n_units),
                relu(inplace=self.inplace),
            ),
        )

        for i in range(2, n_layer + 1):
            self.add_module(
                f"mlp_layer{i}",
                nn.Sequential(
                    nn.Linear(n_units, n_units),
                    nn.LayerNorm(normalized_shape=n_units),
                    relu(inplace=self.inplace),
                ),
            )

    def forward(self, x):
        for i in range(1, self.n_layer + 1):
            x = self.__getattr__(f"mlp_layer{i}")(x)
        return x


class Decoder(nn.Module):
    """
    Decoder.

    Constructor arguments: 
        use_z : (Bool), if True, Decoder will use z as input.
        mlp_units: 512
        mlp_layers: 3
        z_units: 16
        n_harmonics: 101
        n_freq: 65
        gru_units: 512
        bidirectional: False

    input(dict(f0, z(optional), l)) : a dict object which contains key-values below
        f0 : fundamental frequency for each frame. torch.tensor w/ shape(B, time)
        z : (optional) residual information. torch.tensor w/ shape(B, time, z_units)
        loudness : torch.tensor w/ shape(B, time)

        *note dimension of z is not specified in the paper.

    output : a dict object which contains key-values below
        f0 : same as input
        c : torch.tensor w/ shape(B, time, n_harmonics) which satisfies sum(c) == 1
        a : torch.tensor w/ shape(B, time) which satisfies a > 0
        H : noise filter in frequency domain. torch.tensor w/ shape(B, frame_num, filter_coeff_length)
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.batch_size = config.batch_size
        self.max_voices = config.max_voices

        self.pitch_mlp_f0 = MLP(n_input=1, n_units=config.mlp_units, n_layer=config.mlp_layers)
        self.pitch_mlp_loudness = MLP(n_input=1, n_units=config.mlp_units, n_layer=config.mlp_layers)
        self.pitch_mlp_velocity = MLP(n_input=1, n_units=config.mlp_units, n_layer=config.mlp_layers)

        self.noise_mlp_f0 = MLP(n_input=self.max_voices, n_units=config.mlp_units, n_layer=config.mlp_layers)
        self.noise_mlp_velocity = MLP(n_input=self.max_voices, n_units=config.mlp_units, n_layer=config.mlp_layers)
        
        if config.use_z:
            self.mlp_z = MLP(
                n_input=config.z_units, n_units=config.mlp_units, n_layer=config.mlp_layers
            )
            self.num_mlp = 4
        else:
            self.num_mlp = 3


        # GRU and MLP for c encoding
        self.pitch_gru = nn.GRU(
            input_size=self.num_mlp * config.mlp_units,
            hidden_size=config.gru_units,
            num_layers=1,
            batch_first=True,
            bidirectional=config.bidirectional,
        )

        self.pitch_mlp_gru = MLP(
            n_input=config.gru_units * 2 if config.bidirectional else config.gru_units,
            n_units=config.mlp_units,
            n_layer=config.mlp_layers,
            inplace=True,
        )

        # GRU and MLP for a and H encoding\
        self.noise_gru = nn.GRU(
            input_size=self.num_mlp * config.mlp_units,
            hidden_size=config.gru_units,
            num_layers=1,
            batch_first=True,
            bidirectional=config.bidirectional,
        )

        self.noise_mlp_gru = MLP(
            n_input=config.gru_units * 2 if config.bidirectional else config.gru_units,
            n_units=config.mlp_units,
            n_layer=config.mlp_layers,
            inplace=True,
        )

        self.dense_harmonic = nn.Linear(config.mlp_units, config.n_harmonics)
        # one element for overall loudness
        self.dense_filter = nn.Linear(config.mlp_units, config.n_freq + 1)

    def forward(self, batch):
        # Check for smaller validation batches
        self.batch_size = batch["f0"].size(0)

        f0 = batch["f0"]
        pitch_f0 = f0.unsqueeze(-1).reshape(self.batch_size*self.max_voices,f0.size(2), 1)
        noise_f0 = f0.permute(0, 2, 1)

        velocity = batch["velocity"]
        pitch_velocity= velocity.unsqueeze(-1).reshape(self.batch_size * self.max_voices, velocity.size(2), 1)
        noise_velocity = velocity.permute(0, 2, 1)

        loudness = batch['loudness']
        pitch_loudness = loudness.unsqueeze(1).repeat(1,self.max_voices,1)
        pitch_loudness = pitch_loudness.reshape(self.batch_size * self.max_voices, pitch_loudness.size(2)). unsqueeze(-1)
        noise_loudness = loudness.unsqueeze(-1)

        if self.config.use_z:
            z = batch["z"]
            pitch_z = z.unsqueeze(1).repeat(1,self.max_voices,1,1)
            pitch_z = pitch_z.reshape(self.batch_size * self.max_voices, pitch_z.size(2), pitch_z.size(3))
            noise_z = z

            pitch_latent_z = self.mlp_z(pitch_z)
            noise_latent_z = self.mlp_z(noise_z)


        pitch_latent_f0 = self.pitch_mlp_f0(pitch_f0)
        pitch_latent_loudness = self.pitch_mlp_loudness(pitch_loudness)
        pitch_latent_velocity = self.pitch_mlp_velocity(pitch_velocity)

        noise_latent_f0 = self.noise_mlp_f0(noise_f0)
        noise_latent_loudness = self.pitch_mlp_loudness(noise_loudness)
        noise_latent_velocity = self.noise_mlp_velocity(noise_velocity)


        if self.config.use_z:
            pitch_latent = torch.cat((pitch_latent_f0, pitch_latent_velocity, pitch_latent_z, pitch_latent_loudness), dim=-1)
            noise_latent = torch.cat((noise_latent_f0, noise_latent_velocity, noise_latent_z, noise_latent_loudness), dim=-1)
        else:
            pitch_latent = torch.cat((pitch_latent_f0, pitch_latent_velocity, pitch_latent_loudness), dim=-1)
            noise_latent = torch.cat((noise_latent_f0, noise_latent_velocity, noise_latent_loudness), dim=-1)

        pitch_latent = self.pitch_gru(pitch_latent)[0]
        pitch_latent = self.pitch_mlp_gru(pitch_latent)

        noise_latent = self.noise_gru(noise_latent)[0]
        noise_latent = self.noise_mlp_gru(noise_latent)

        # Reconstruct polyphony channel
        pitch_latent = pitch_latent.reshape(self.batch_size, self.max_voices, pitch_latent.size(1), pitch_latent.size(2))

        amplitude = self.dense_harmonic(pitch_latent)

        # a = torch.sigmoid(amplitude[..., 0])
        c = F.softmax(amplitude, dim=-1)
        # c = c[:,0,...].unsqueeze(1).repeat(1,self.max_voices,1,1)

        noise_amp = self.dense_filter(noise_latent)
        H = Decoder.modified_sigmoid(noise_amp[..., 1:])

        a = Decoder.modified_sigmoid(noise_amp[..., 0])
        # H = H.mean(dim = 1,keepdim = False)
        # a = a.mean(dim = 1,keepdim = False)
        # H = H[:,0,...]
        # a = a[:,0,...]
        c = c.permute(0, 1, 3, 2)  # to match the shape of harmonic oscillator's input.

        return dict(f0=batch["f0"], v=batch["velocity"], a=a, c=c, H=H)

    @staticmethod
    def modified_sigmoid(a):
        a = a.sigmoid()
        a = a.pow(2.3026)  # log10
        a = a.mul(2.0)
        a.add_(1e-7)
        return a

