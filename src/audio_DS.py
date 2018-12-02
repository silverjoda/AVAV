import numpy as np
from os import listdir
import torch as T
import torchaudio
import matplotlib.pyplot as plt
import os

class AudioSet:
    def __init__(self):
        self.path = 'audio_sources/music'

        # Read audio files and make numpy dataset
        self.dataset = self.make_dataset()

        print("Read dataset consisting of {} samples".format(len(self.dataset)))


    def make_dataset(self):
        '''

        Returns
        -------
        Numpy array dataset of video frames from all video files. Audio is int16 : [-32768:32768]

        '''

        # Frames will be stored here
        samplelist = []

        # Check all available files
        files = listdir(self.path)

        for f in files:
            full_f = os.path.join(self.path, f)

            # Data is [len x num_channels] (num channels = 2 if stereo, 1 if mono)
            data, sample_rate = torchaudio.load(full_f, normalization=True)

            # If sample is mono then duplicate channel
            if data.shape[1] == 1:
                data = data.repeat(1,2)

            # Transpose data to have dimensions [channels, length]
            samplelist.append(data)

        return T.cat(samplelist)


    def get_cons_sample(self, N):
        '''

        Parameters
        ----------
        N - sample size, res - resolution

        Returns
        -------
        N consecutive samples (individual data points) from the audio dataset
        '''

        # Length of dataset
        d_len = len(self.dataset)

        # Get random starting point
        rnd_pt = np.random.randint(0, d_len - N + 1)  # The +1 is due to numpy.random

        # Get sample
        sample = self.dataset[rnd_pt: rnd_pt + N]

        # Normalize and turn to float
        #norm_sample_float = (sample.astype(np.float32) / (32767. / 2)) - 1

        return sample.transpose(1,0)


    def get_cons_batch(self, N, batchsize):
        samples = []
        for _ in range(batchsize):
            samples.append(self.get_cons_sample(N))
        return T.stack(samples)


    def writetofile(self, data, fname, rate):
        torchaudio.save(fname, data, rate)  # saves tensor to file


    def writetostream(self, data):
        raise NotImplementedError


if __name__ == "__main__":
    # DEBUG#
    #reader = AudioSet()

    sound, sample_rate = torchaudio.load('audio_sources/music/audio_4.mp3', normalization=True)
    print(sound.min(), sound.max())
    #torchaudio.save('foo_save.mp3', sound, sample_rate)  # saves tensor to file
