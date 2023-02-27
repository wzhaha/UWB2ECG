# data augmentation
import numpy as np
import random
from scipy import signal
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from utils.datautil import mapminmax


def scale_data(data, scale_rate):
    ori_len = len(data)
    scale_len = int(ori_len*scale_rate)
    new_data = signal.resample(data, scale_len, axis=0)
    clip_data = new_data[:ori_len]
    return clip_data


def add_sine(data):
    t = np.arange(0, 10, 1/200)
    amp = (max(data)-min(data))/8
    frquency = random.randrange(15, 25, 1)/100

    sine_data = amp*np.sin(2*np.pi*frquency*t)
    new_data = data + sine_data
    new_data = (new_data - np.mean(new_data)) / np.std(new_data)
    new_data = mapminmax(new_data)
    return new_data


def data_augmentation():
    ori_data_path = './data/train_uwb.npy'
    data = np.load(ori_data_path)
    data_len = data.shape[0]
    print('ori data shape:', data.shape)

    aug_data = []
    for index in range(data_len):
        ecg = data[index, :, 0]
        amplitude_data = data[index, :, 1]
        phase_data = data[index, :, 2]

        # scale data
        scale_rate = random.randrange(100, 120, 1)/100
        scale_ecg = scale_data(ecg, scale_rate)
        scale_amp = scale_data(amplitude_data, scale_rate)
        scale_pha = scale_data(phase_data, scale_rate)
        temp = np.stack((scale_ecg, scale_amp, scale_pha)).T
        aug_data.append(temp)

        # add sine, only add to phase or amplitude,
        # random_sel=1: amp, random_sel=2: phase, random_sel=3: all
        sine_amp = amplitude_data
        sine_pha = phase_data
        random_sel = random.randint(1, 3)
        if random_sel == 1:
            sine_amp = add_sine(sine_amp)
        elif random_sel == 2:
            sine_pha = add_sine(sine_pha)
        else:
            sine_amp = add_sine(sine_amp)
            sine_pha = add_sine(sine_pha)
        temp = np.stack((ecg, sine_amp, sine_pha)).T
        aug_data.append(temp)
    aug_data = np.asarray(aug_data)

    all_data = np.concatenate((data, aug_data), axis=0)

    np.save("data/train_uwb_aug.npy", all_data)
    print('new data shape:', all_data.shape)


if __name__ == '__main__':
    data_augmentation()
