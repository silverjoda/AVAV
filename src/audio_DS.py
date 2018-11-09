import numpy as np
from os import listdir
from scipy.io import wavfile
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
            samplerate, data = wavfile.read(full_f)
            samplelist.append(data)

        return np.concatenate(samplelist, axis=0)


    def get_cons_sample(self, N):
        '''

        Parameters
        ----------
        N - sample size, res - resolution

        Returns
        -------
        N consecutive frames from the video dataset
        '''

        # Length of dataset
        d_len = len(self.dataset)

        # Get random starting point
        rnd_pt = np.random.randint(0, d_len - N + 1)  # The +1 is due to numpy.random

        # Get sample
        sample = self.dataset[rnd_pt: rnd_pt + N]

        # Normalize and turn to float
        norm_sample_float = (sample.astype(np.float32) / (32767. / 2)) - 1

        return norm_sample_float


    def get_cons_batch(self, N, batchsize):
        pass


    def writetofile(self, data, fname, rate):
        data = data / np.max(data)
        data = data * 2 - np.max(data)
        print(data.shape)
        plt.plot(data[:44111])
        #plt.show()
        print("Writing data type: {}, min: {}, max: {}".format(data.dtype, np.min(data), np.max(data)))
        wavfile.write(os.path.join(self.path, fname), rate, data)

    def writetostream(self, data):
        raise NotImplementedError


if __name__ == "__main__":
    # DEBUG#
    #reader = AudioSet()

    samplerate, data = wavfile.read('audio_sources/music/audio_1.wav')
    print(np.max(data));print(np.min(data))

    # data = np.random.randn(20000)
    # wavfile.write("shitfile.wav", 44100,data)