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
        self.max_voices = 0 #config.max_voices

        self.mlp_f0 = MLP(n_input=1, n_units=config.mlp_units, n_layer=config.mlp_layers)
        self.mlp_loudness = MLP(n_input=1, n_units=config.mlp_units, n_layer=config.mlp_layers)
        self.mlp_velocity = MLP(n_input=1, n_units=config.mlp_units, n_layer=config.mlp_layers)
        if config.use_z:
            self.mlp_z = MLP(
                n_input=config.z_units, n_units=config.mlp_units, n_layer=config.mlp_layers
            )
            self.num_mlp = 4
        else:
            self.num_mlp = 3

        self.gru = nn.GRU(
            input_size=self.num_mlp * config.mlp_units,
            hidden_size=config.gru_units,
            num_layers=1,
            batch_first=True,
            bidirectional=config.bidirectional,
        )

        self.mlp_gru = MLP(
            n_input=config.gru_units * 2 if config.bidirectional else config.gru_units,
            n_units=config.mlp_units,
            n_layer=config.mlp_layers,
            inplace=True,
        )

        # one element for overall loudness
        self.dense_harmonic = nn.Linear(config.mlp_units, config.n_harmonics + 1)
        self.dense_filter = nn.Linear(config.mlp_units, config.n_freq)

    def forward(self, batch):
        f0 = batch["f0"].unsqueeze(-1)

        # Collapse polyphony channels into batch dimension
        self.batch_size = f0.size(0)
        self.max_voices = f0.size(1)

        f0 = f0.reshape(f0.size(0)*f0.size(1), f0.size(2), f0.size(3))

        velocity = batch["velocity"].unsqueeze(-1)
        velocity = velocity.reshape(velocity.size(0)*velocity.size(1), velocity.size(2), velocity.size(3))
        
        loudness = batch['loudness'].unsqueeze(1).repeat(1,self.max_voices,1).unsqueeze(-1)
        loudness = loudness.reshape(loudness.size(0)*loudness.size(1), loudness.size(2), loudness.size(3))

        if self.config.use_z:
            z = batch["z"].unsqueeze(1).repeat(1,self.max_voices,1,1)
            z= z.reshape(z.size(0)*z.size(1), z.size(2), z.size(3))
            latent_z = self.mlp_z(z)
       

        latent_f0 = self.mlp_f0(f0)
        latent_loudness = self.mlp_loudness(loudness)
        latent_velocity = self.mlp_velocity(velocity)

        if self.config.use_z:
            latent = torch.cat((latent_f0, latent_velocity, latent_z, latent_loudness), dim=-1)
        else:
            latent = torch.cat((latent_f0, latent_velocity, latent_loudness), dim=-1)

        latent, (h) = self.gru(latent)
        latent = self.mlp_gru(latent)

        # Reconstruct polyphony channel
        latent = latent.reshape(self.batch_size, self.max_voices, latent.size(1), latent.size(2))

        amplitude = self.dense_harmonic(latent)

        a = amplitude[..., 0]
        a = Decoder.modified_sigmoid(a)

        # a = torch.sigmoid(amplitude[..., 0])
        c = F.softmax(amplitude[..., 1:], dim=-1)
        c = c[:,0,...].unsqueeze(1).repeat(1,self.max_voices,1,1)

        H = self.dense_filter(latent)
        H = Decoder.modified_sigmoid(H)

        #H = H.mean(dim = 1,keepdim = False)
        #a = a.mean(dim = 1,keepdim = False)
        H = H[:,0,...]
        a = a[:,0,...]
        c = c.permute(0, 1, 3, 2)  # to match the shape of harmonic oscillator's input.

        return dict(f0=batch["f0"], v=batch["velocity"], a=a, c=c, H=H)

    @staticmethod
    def modified_sigmoid(a):
        a = a.sigmoid()
        a = a.pow(2.3026)  # log10
        a = a.mul(2.0)
        a.add_(1e-7)
        return a

