from video_DS import VidSet
from networks import AVA
from audio_DS import AudioSet
import numpy as np
import cv2
import torch as T

def main():

    LOAD = False

    imres = 128
    video_rate = 60
    audio_rate = 44100
    iters = 20000
    batchsize = 64

    # Make dataset
    video_reader = VidSet(imres)
    audio_reader = AudioSet()

    # Make network
    network = AVA(audio_rate, video_rate).cuda()
    #network = T.load("trained_networks/ava.pt")

    # Train ==============
    network.fit(audio_reader, iters, batchsize)
    T.save(network, "trained_networks/ava.pt")

    # Test ===============
    secs = 10
    N_frames = 10 * 60
    test_sample = audio_reader.get_cons_sample(audio_rate * secs).cuda()
    frames = []
    step =  int(audio_rate / video_rate)
    for i in range(N_frames):
        sample = test_sample[:, i * step : i * step + step]
        enc = network.encode(sample.unsqueeze(0))
        frames.append(enc)

    vid_encoding = T.cat(frames, 0).permute((0, 2, 3, 1))
    video_reader.write_video(vid_encoding.detach().cpu().numpy(), test_sample.detach().cpu().numpy(), "test_encoding")

    # TODO: Compare reconstruction and ground truth by MSE, PLOT and FFT
    # TODO: ADD AUDIO TO VIDEO OUTPUT


if __name__ == "__main__":
    main()