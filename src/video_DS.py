import numpy as np
from os import listdir
import cv2
import os


class VidSet:
    def __init__(self, res, viz=False):
        # Video files are kept here
        self.path = 'video_sources'

        # Maximum amount of frames
        self.max_frames = 24 * 1000

        # Video frame resolution
        self.res = res

        # Visualize
        self.viz = viz

        # Read video files and make numpy dataset
        self.dataset = self.make_dataset()

        print("Read video dataset consisting of {} frames".format(len(self.dataset)))


    def make_dataset(self):
        '''

        Returns
        -------
        Numpy array dataset of video frames from all video files

        '''

        # Frames will be stored here
        framelist = []

        # Check all available files
        files = listdir(self.path)

        f_ctr = 0

        for f in files:
            full_f = os.path.join(self.path, f)
            cap = cv2.VideoCapture(full_f)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                if f_ctr > self.max_frames - 1:
                    break
                f_ctr += 1

                # Raw frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Resized frame
                gray_rs = cv2.resize(gray, (self.res, self.res))

                # Add frame to list
                framelist.append(gray_rs)

                # Visualize dataset
                if self.viz:
                    cv2.imshow('frame', gray_rs)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            cap.release()
            cv2.destroyAllWindows()

        return np.array(framelist)


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
        rnd_pt = np.random.randint(0, d_len - N + 1) # The +1 is due to numpy.random

        # Get sample
        sample = self.dataset[rnd_pt : rnd_pt + N]

        # Normalize and turn to float
        norm_sample_float = (sample.astype(np.float32) / (255. / 2)) - 1

        return norm_sample_float


    def get_rnd_sample(self, N):
        '''

        Parameters
        ----------
        N - sample size, res - resolution

        Returns
        -------
        N random frames from the video dataset
        '''

        # Length of dataset
        d_len = len(self.dataset)

        # Choice arr
        choice_arr = np.random.choice(d_len, N, replace=False)

        # Chosen sample
        return self.dataset[choice_arr]


    def write_video(self, data, name):
        if len(data.shape) == 3:
            print("Data is GS, need to converting to rgb")
            data = np.repeat(data[:, :, :, np.newaxis], 3, axis=3)

        video = cv2.VideoWriter('{}.avi'.format(name), cv2.VideoWriter_fourcc(*"MJPG"), 60, (self.res, self.res))
        for d in data:
            video.write(d)
        video.release()


if __name__ == "__main__":
    #DEBUG#
    reader = VidSet(128)

    dsize = len(reader.dataset)

    cons_sample = reader.get_cons_sample(1)
    cons_sample = reader.get_cons_sample(5)
    cons_sample = reader.get_cons_sample(dsize)

    rnd_sample = reader.get_rnd_sample(1)
    rnd_sample = reader.get_rnd_sample(5)
    rnd_sample = reader.get_rnd_sample(dsize)

    # writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 480))
    # for frame in range(1000):
    #     writer.write(np.random.randint(0, 255, (480, 640, 3)).astype('uint8'))
    # writer.release()
    reader.write_video(np.random.randint(0, 255, (300, 128, 128, 3)).astype('uint8'), 'rndvid')

    pass










