from video_DS import VidSet
from networks import AVA
from audio_DS import AudioSet
import numpy as np
import cv2
import torch as T

def main():

    LOAD = False

    imres = 128
    FPS = 60
    samplerate = 44100
    n_iters = 20000
    batchsize = 32

    # Make dataset
    video_reader = VidSet(imres)
    audio_reader = AudioSet()

    # Make network
    network = AVA("testnet", samplerate, FPS)
    #network = T.load("trained_networks/ava.pt")

    # Train ==============
    network.train()

    # Test ===============
    test_sample = audio_reader.get_cons_sample(3)
    reconstruction = network.reconstruct(test_sample)
    assert test_sample.shape == reconstruction.shape, print("Reconstruction does not match initial video")

    vid_encoding = network.encode(test_sample)
    video_reader.write_video(vid_encoding, "test_encoding")



if __name__ == "__main__":
    main()