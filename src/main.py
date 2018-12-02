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

    exit()

    # Test ===============
    test_sample = audio_reader.get_cons_sample(audio_rate * 3).unsqueeze(0)
    reconstruction = network.reconstruct(test_sample)
    assert test_sample.shape == reconstruction.shape, print("Reconstruction does not match initial input")

    # TODO: Compare reconstruction and ground truth by MSE, PLOT and FFT

    vid_encoding = network.encode(test_sample)
    video_reader.write_video(vid_encoding, "test_encoding")


if __name__ == "__main__":
    main()