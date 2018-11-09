import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


class VAV:
    def __init__(self, name, res, rate, FPS):
        self.name = name
        self.res = res
        self.rate = rate
        self.FPS = FPS
        self.sample_per_frame = self.rate // self.FPS
        self.bottleneck_size = 550
        self.weights_path = 'models/{}'.format(name)


class AVA(nn.Module):
    def __init__(self, name, rate, FPS):
        super(AVA, self).__init__()

        # Class parameters
        self.name = name
        self.rate = rate
        self.FPS = FPS
        self.sample_per_frame = self.rate // self.FPS

        # Neural network parameters:

        # Audio conv, 1470 audio input samples, 2 channels
        self.a_conv1 = nn.Conv1d(2, 8, kernel_size=5, stride=5, dilation=2)
        self.a_conv2 = nn.Conv1d(8, 16, kernel_size=3, stride=3, dilation=1)
        self.a_conv3 = nn.Conv1d(16, 1, kernel_size=3, stride=3, dilation=1)

        # Video deconv from audio
        self.v_deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1)
        self.v_deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=3, dilation=1)
        self.v_deconv3 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=3, dilation=1)
        self.v_deconv4 = nn.ConvTranspose2d(8, 3, kernel_size=3, stride=3, dilation=1)

        # Video enc back to audio
        self.v_conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=3, dilation=1)
        self.v_conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=3, dilation=1)
        self.v_conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=3, dilation=1)
        self.v_conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, dilation=0)

        # Audio deconv (output 1470 audio samples, 2 channels)
        self.a_deconv1 = nn.ConvTranspose1d(1, 8, kernel_size=3, stride=3, dilation=1)
        self.a_deconv2 = nn.ConvTranspose1d(8, 16, kernel_size=3, stride=3, dilation=1)
        self.a_deconv3 = nn.ConvTranspose1d(16, 16, kernel_size=5, stride=3, dilation=1)
        self.a_deconv4 = nn.ConvTranspose1d(16, 2, kernel_size=5, stride=2, dilation=0)


    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        # Audio conv, 1470 audio input samples, 2 channels
        x = F.relu(self.a_conv1(x))
        x = F.relu(self.a_conv2(x))
        x = F.relu(self.a_conv3(x))

        # Video deconv from audio
        x = F.relu(self.v_deconv1(x))
        x = F.relu(self.v_deconv2(x))
        x = F.relu(self.v_deconv3(x))
        x = F.relu(self.v_deconv4(x))

        return nn.upsample(x, size=(128, 128, 3))

    def decode(self, x):
        # Video enc back to audio
        x = F.relu(self.v_conv1(x))
        x = F.relu(self.v_conv2(x))
        x = F.relu(self.v_conv3(x))
        x = F.relu(self.v_conv4(x))

        # Audio deconv (output 1470 audio samples, 2 channels)
        x = F.relu(self.a_deconv1(x))
        x = F.relu(self.a_deconv2(x))
        x = F.relu(self.a_deconv3(x))
        x = F.relu(self.a_deconv4(x))

        x = nn.upsample(x, size=(1470))

        return x

    def fit(self, DS, iters, batchsize):
        reconstruction_loss = torch.nn.MSELoss()
        optim = torch.optim.Adam(self.parameters(), lr=3e-4)

        for i in range(iters):
            # Get batch of data
            data = DS.get_cons_batch(batchsize)

            # Perform forward pass
            recon = self.forward(data)

            # Calculate loss
            loss = reconstruction_loss(data, recon)

            # Perform backward pass
            loss.backward()

            # Step optimization
            optim.zero_grad()
            optim.step()
