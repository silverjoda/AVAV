from video_DS import VidSet
from networks import AVA
from audio_DS import AudioSet
import numpy as np
import cv2


def main():

    LOAD = False

    imres = 128
    FPS = 30
    samplerate = 16384
    n_iters = 20000
    batchsize = 16

    # Make network
    network = AVA("testnet", samplerate, FPS)

    # Make dataset
    video_reader = VidSet(imres)

    if LOAD:
        # Load weights
        network.restore_weights()
    else:
        # Train
        for i in range(n_iters):
            batch = video_reader.get_rnd_sample(batchsize)
            mse = network.train(batch)

            if i % 10 == 0:
                print("Iteration {}/{}, mse: {}".\
                    format(i, n_iters, mse))

        # Save weights
        network.save_weights()

    # Test ===============
    test_sample = video_reader.get_cons_sample(300)
    reconstruction = network.reconstruct(test_sample)
    assert test_sample.shape == reconstruction.shape, print("Reconstruction does not match initial video")
    combined_video = np.concatenate((test_sample, reconstruction), axis=2)

    writer = cv2.VideoWriter('video_results/test1.avi', cv2.VideoWriter_fourcc(*'PIM1'), 25, (2 * imres, imres), False)
    for c in combined_video:
        frame = c.astype(np.uint8)
        #x = np.random.randint(255, size=(imres, 2 * imres)).astype('uint8')
        writer.write(frame)

    audio_encoding = network.encode(test_sample)
    audio_encoding = np.reshape(audio_encoding, [-1])

    audio_writer = Audiowriter()
    audio_writer.writetofile(audio_encoding,
                             "firsttest.wav",
                             samplerate)



if __name__ == "__main__":
    main()