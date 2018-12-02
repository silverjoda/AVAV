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
    def __init__(self, audio_rate, video_rate):
        super(AVA, self).__init__()

        # Class parameters
        self.audio_rate = audio_rate
        self.video_rate = video_rate
        self.samples_per_frame = self.audio_rate // self.video_rate

        # Neural network parameters:

        # Audio conv, 1470 audio input samples, 2 channels
        self.a_conv1 = nn.Conv1d(2, 16, kernel_size=5, stride=5, dilation=2)
        self.a_conv2 = nn.Conv1d(16, 16, kernel_size=3, stride=2, dilation=1)
        self.a_conv3 = nn.Conv1d(16, 16, kernel_size=3, stride=2, dilation=1)

        # Video deconv from audio
        self.v_deconv1 = nn.ConvTranspose2d(35, 32, kernel_size=3, stride=1, dilation=1)
        self.v_deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, dilation=1)
        self.v_deconv3 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, dilation=1)
        self.v_deconv4 = nn.ConvTranspose2d(8, 3, kernel_size=3, stride=1, dilation=1)

        # Video enc back to audio
        self.v_conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, dilation=1)
        self.v_conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, dilation=1)
        self.v_conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, dilation=1)
        self.v_conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, dilation=1)

        # Audio deconv (output 1470 audio samples, 2 channels)
        self.a_deconv1 = nn.ConvTranspose1d(36, 32, kernel_size=3, stride=1, dilation=1)
        self.a_deconv2 = nn.ConvTranspose1d(32, 16, kernel_size=3, stride=1, dilation=1)
        self.a_deconv3 = nn.ConvTranspose1d(16, 16, kernel_size=5, stride=1, dilation=1)
        self.a_deconv4 = nn.ConvTranspose1d(16, 2, kernel_size=5, stride=1, dilation=1, padding=2)

        torch.nn.init.xavier_uniform_(self.a_conv1.weight)
        torch.nn.init.xavier_uniform_(self.a_conv2.weight)
        torch.nn.init.xavier_uniform_(self.a_conv3.weight)

        torch.nn.init.xavier_uniform_(self.v_deconv1.weight)
        torch.nn.init.xavier_uniform_(self.v_deconv2.weight)
        torch.nn.init.xavier_uniform_(self.v_deconv3.weight)
        torch.nn.init.xavier_uniform_(self.v_deconv4.weight)

        torch.nn.init.xavier_uniform_(self.v_conv1.weight)
        torch.nn.init.xavier_uniform_(self.v_conv2.weight)
        torch.nn.init.xavier_uniform_(self.v_conv3.weight)
        torch.nn.init.xavier_uniform_(self.v_conv4.weight)

        torch.nn.init.xavier_uniform_(self.a_deconv1.weight)
        torch.nn.init.xavier_uniform_(self.a_deconv2.weight)
        torch.nn.init.xavier_uniform_(self.a_deconv3.weight)
        torch.nn.init.xavier_uniform_(self.a_deconv4.weight)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


    def encode(self, x):
        # Audio conv, 1470 audio input samples, 2 channels
        x = F.relu(self.a_conv1(x))
        x = F.relu(self.a_conv2(x))
        x = F.relu(self.a_conv3(x))

        # Reshape x
        x = x.view(x.shape[0], 35, 4, 4)

        # Video deconv from audio
        x = F.upsample_bilinear(F.relu(self.v_deconv1(x)), scale_factor=2)
        x = F.upsample_bilinear(F.relu(self.v_deconv2(x)), scale_factor=2)
        x = F.upsample_bilinear(F.relu(self.v_deconv3(x)), scale_factor=2)
        x = F.upsample_bilinear(F.tanh(self.v_deconv4(x)), size=128)

        return x


    def decode(self, x):
        # Video enc back to audio
        x = F.avg_pool2d(F.relu(self.v_conv1(x)), (2,2))
        x = F.avg_pool2d(F.relu(self.v_conv2(x)), (2,2))
        x = F.avg_pool2d(F.relu(self.v_conv3(x)), (2,2))
        x = F.avg_pool2d(F.relu(self.v_conv4(x)), (2,2))

        # Reshape
        x = x.view(x.shape[0], 36, 64)

        x = F.upsample(F.relu(self.a_deconv1(x)), scale_factor=2)
        x = F.upsample(F.relu(self.a_deconv2(x)), scale_factor=2.5)
        x = F.upsample(F.tanh(self.a_deconv3(x)), size=(self.samples_per_frame))
        x = self.a_deconv4(x)

        return x


    def fit(self, DS, iters, batchsize):
        reconstruction_loss = torch.nn.MSELoss()
        optim = torch.optim.Adam(self.parameters(), lr=3e-4, weight_decay=0.001)

        for i in range(iters):
            # Get batch of data
            data = DS.get_cons_batch(self.samples_per_frame, batchsize).cuda()

            # Perform forward pass
            recon = self.forward(data)

            # Calculate loss
            loss = reconstruction_loss(data, recon)

            # Perform backward pass
            optim.zero_grad()
            loss.backward()

            # Step optimization
            optim.step()

            if i % 10 == 0:
                print("Iteration {}/{}, loss: {}".format(i,iters,loss))
