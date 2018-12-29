from video_DS import VidSet
from networks import *
from audio_DS import AudioSet
import numpy as np
import cv2
import torch as T

def main():

    LOAD = False

    imres = 128
    video_rate = 60
    audio_rate = 44100
    iters = 1000
    batchsize = 64

    # Make dataset
    video_reader = VidSet(imres)
    audio_reader = AudioSet()

    # Make network
    network = AVA_PW(audio_rate, video_rate).cuda()
    #network = T.load("trained_networks/ava.pt")

    # Train ==============
    network.fit(audio_reader, iters, batchsize)
    T.save(network, "trained_networks/aAtva.pt")

    # Test ===============
    secs = 30
    N_frames = secs * 60
    test_sample = audio_reader.get_cons_sample(audio_rate * secs).cuda()
    frames = []
    recon_samples = []
    step = int(audio_rate / video_rate)
    for i in range(N_frames):
        if i == N_frames - 1:
            sample = T.cat((test_sample[:, i * step: i * step + step], T.zeros(2, step).cuda()), 1)
        else:
            sample = test_sample[:, i * step : i * step + 2 * step]

        with T.no_grad():
            enc = network.encode(sample.unsqueeze(0), single=True)
            recon = network.forward(sample.unsqueeze(0))
            frames.append(enc)
            recon_samples.append(recon[:, :, :step])

    audio_encoding = T.cat(recon_samples, 2)[0]

    # Write audio to file
    audio_reader.writetofile(test_sample.transpose(0,1).cpu(), "test_audio.mp3", audio_rate)
    audio_reader.writetofile(audio_encoding.transpose(0,1).cpu(), "encoding_audio.mp3", audio_rate)

    # Write video to file
    vid_encoding = T.cat(frames, 0).permute((0, 2, 3, 1))
    video_reader.write_video(vid_encoding.detach().cpu().numpy(), test_sample.detach().cpu().numpy(), "test_video_encoding")

    # TODO: Compare reconstruction and ground truth by MSE, PLOT and FFT
    # TODO: Save original audio sample, reconstruction and audio/video layover



if __name__ == "__main__":
    main()