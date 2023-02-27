import scipy.io as sio
import numpy as np
from torch.utils.data import Dataset
import torch
from scipy import signal
import os
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import random


def load_data(opt):
    print("start to load data")
    if opt.dataset == 'uwbdata':
        ecgs = []
        win_amplitude_datas = []
        win_phase_datas = []
        # 0, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21
        data_paths = ['./data/uwb_data/%d.mat' % (i) for i in [0, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21]]
        for data_path in data_paths:
            data = sio.loadmat(data_path)
            file_ecg = data['ecgs']
            file_amp_data = data['amplitudes']
            file_pha_data = data['phases']

            for index in range(file_ecg.shape[0]):
                # remove none
                file_ecg[index, :] = remove_nan(file_ecg[index, :])
                # z_score
                file_ecg[index, :] = (file_ecg[index, :] - np.mean(file_ecg[index, :])) / np.std(file_ecg[index, :])
                file_amp_data[index, :] = (file_amp_data[index, :] - np.mean(file_amp_data[index, :])) / np.std(file_amp_data[index, :])
                file_pha_data[index, :] = (file_pha_data[index, :] - np.mean(file_pha_data[index, :])) / np.std(file_pha_data[index, :])

                # map to [-1, 1]
                if opt.sample_len == 2000:
                    # ecgs.append(file_ecg[index, :])
                    # win_amplitude_datas.append(file_amp_data[index, :])
                    # win_phase_datas.append(file_pha_data[index, :])
                    ecgs.append(mapminmax(file_ecg[index, :]))
                    win_amplitude_datas.append(mapminmax(file_amp_data[index, :]))
                    win_phase_datas.append(mapminmax(file_pha_data[index, :]))
                elif opt.sample_len == 1000:
                    ecgs.append(file_ecg[index, :1000])
                    win_amplitude_datas.append(file_amp_data[index, :1000])
                    win_phase_datas.append(file_pha_data[index, :1000])
                    ecgs.append(file_ecg[index, 1000:])
                    win_amplitude_datas.append(file_amp_data[index, 1000:])
                    win_phase_datas.append(file_pha_data[index, 1000:])

        print("end to load data")
        return np.asarray(ecgs), np.asarray(win_amplitude_datas), np.asarray(win_phase_datas)
    elif opt.dataset == 'nature_data':
        win_ecgs = []
        win_amplitudes = []
        win_phases = []
        #
        # # add shi xing zao bo
        # data_path = '/home/wz/桌面/UWB2ECG/data/nature_data_resting_for_train/patient_data/GDN0001_1_Resting.mat'
        # data = sio.loadmat(data_path)
        # file_ecg = data['ecg_data']
        # file_amp_data = data['amplitude_data']
        # file_pha_data = data['angle_data']
        # for index in range(file_ecg.shape[1]):
        #     # resample
        #     win_ecg = signal.resample(file_ecg[:, index], 2000, axis=0)
        #     win_amplitude = -signal.resample(file_amp_data[:, index], 2000, axis=0)
        #     win_phase = signal.resample(file_pha_data[:, index], 2000, axis=0)
        #
        #     # z_score
        #     ecg = (win_ecg - np.mean(win_ecg)) / np.std(win_ecg)
        #     amplitude = (win_amplitude - np.mean(win_amplitude)) / np.std(win_amplitude)
        #     phase = (win_phase - np.mean(win_phase)) / np.std(win_phase)
        #
        #     win_ecgs.append(mapminmax(ecg))
        #     win_amplitudes.append(mapminmax(amplitude))
        #     win_phases.append(mapminmax(phase))
        #
        #     # win_ecgs.append(ecg)
        #     # win_amplitudes.append(amplitude)
        #     # win_phases.append(phase)

        dir_path = './data/nature_data_resting_for_train/ori_data/'
        file_names = sorted(os.listdir(dir_path))
        if '.DS_Store' in file_names:
            file_names.remove('.DS_Store')
        file_names = [i for i in file_names if 'GDN' in i]
        if opt.except_file is not None:
            file_names = [i for i in file_names if opt.except_file not in i]
        if opt.only_file is not None:
            file_names = [i for i in file_names if opt.only_file in i]

        win_step = 2
        win_len = 10

        for file_name in file_names:
            data_path = os.path.join(dir_path, file_name)
            data = sio.loadmat(data_path)
            fs_ecg = data['fs_ecg'][0, 0]
            fs_radar = data['fs_radar'][0, 0]
            ecgs = data['ecg']
            amplitude_datas = data['amplitude_data']
            angle_datas = data['angle_data']

            time_len = int(ecgs.shape[0]/fs_ecg)

            for time_index in range(0, time_len-win_len, win_step):
                win_ecg = ecgs[time_index*fs_ecg:fs_ecg*(time_index+win_len)]
                win_amplitude = amplitude_datas[time_index * fs_ecg:fs_ecg * (time_index + win_len)]
                win_phase = angle_datas[time_index * fs_ecg:fs_ecg * (time_index + win_len)]

                win_ecg = remove_nan(win_ecg)

                # resample
                win_ecg = signal.resample(win_ecg, 2000, axis=0)
                win_amplitude = signal.resample(win_amplitude, 2000, axis=0)
                win_phase = signal.resample(win_phase, 2000, axis=0)

                # 預處理一下
                win_ecg = (win_ecg - np.mean(win_ecg)) / np.std(win_ecg)
                win_amplitude = (win_amplitude - np.mean(win_amplitude)) / np.std(win_amplitude)
                win_phase = (win_phase - np.mean(win_phase)) / np.std(win_phase)

                win_ecgs.append(mapminmax(win_ecg[:, 0]))
                win_amplitudes.append(mapminmax(win_amplitude[:, 0]))
                win_phases.append(mapminmax(win_phase[:, 0]))

                # win_ecgs.append((win_ecg[:, 0]))
                # win_amplitudes.append((win_amplitude[:, 0]))
                # win_phases.append((win_phase[:, 0]))

        print("end to load data")
        return np.asarray(win_ecgs), np.asarray(win_amplitudes), np.asarray(win_phases)


def pre_process(ECG, fs):
    # 貸通濾波
    fmin = 1
    f_max = 10
    fminn = fmin / (fs / 2)
    fmaxn = f_max / (fs / 2)
    [b, a] = signal.butter(1, [fminn, fmaxn], 'band')
    ECG = signal.filtfilt(b, a, ECG, axis=0)

    # 去除基線
    base_trend = medfilt1mit(ECG, 101)
    # base_trend = signal.medfilt(ECG.reshape(-1), 101).reshape((-1, 1))
    ECG = ECG - base_trend


    return ECG

def medfilt1mit(x, m):
    n = len(x)
    m2 = int(m/2)
    xi = np.median(x[0:min(n, m)])
    xf = np.median(x[n-1 - min(n, m) + 1:])
    xt = np.concatenate((xi+np.zeros([m2, 1]), x, xf+np.zeros([m2, 1])))
    xt = signal.medfilt(xt.flatten(), m)

    xmf = xt[m2:-m2]
    return xmf.reshape((-1, 1))


def remove_nan(data):
    NaN_res = np.isnan(data)
    if True in NaN_res:
        pos = np.where(NaN_res == True)
        if len(pos) > 1:
            data_index = pos[0]
            channel_index = pos[1]
            for i, j in zip(data_index, channel_index):
                data[i, j] = data[i - 1, j]
    return data


def mapminmax(x, ymin=-1, ymax=1):
    try:
        out = (ymax-ymin)*(x-min(x))/(max(x)-min(x))+ymin
    except:
        return x
    return out


def meanMiMaSc(v, nel, percmi, percma):
    """
    Compute the average value of the minima and of the maxima computed on data intervals of the input vector.
    Distribution tails can be excluded using parameters "perci" and "percf"
    :param v:input data vector
    :param nel:data interval length
    :param percmi: of min values (mins and maxs) to be excluded (if positive),number of min values (if negative)
    :param percma: of max values to be excluded (if positive), number of max (mins and maxs) values (if negative)
    :return:
    """

    mini = np.zeros(int(len(v) / nel))
    maxi = np.zeros(int(len(v) / nel))

    j = 0
    for i in range(0, len(v)-nel, nel):
        mini[j] = min(v[i:i + nel])
        maxi[j] = max(v[i:i + nel])
        j = j + 1

    if percmi < 0:
        ii = 1-percmi
    else:
        ii = int(len(maxi) * percmi / 100)

    if percma < 0:
        fi = len(maxi)+percma
    else:
        fi = len(maxi)-int(len(maxi)*percma/100)

    omaxi = np.sort(maxi)
    omini = np.sort(mini)
    meaMi = np.mean(omini[ii:fi])
    meaMa = np.mean(omaxi[ii:fi])

    return meaMi, meaMa


def trainTestSplit(ecgs, imf5s, imf6s, trainPercent):
    data_len = ecgs.shape[0]
    ids = np.random.permutation(data_len)
    train_ids = ids[:int(data_len*trainPercent)]
    test_ids = ids[int(data_len*trainPercent):]

    ecgs_train = np.expand_dims(ecgs[train_ids],2)
    ecgs_test = np.expand_dims(ecgs[test_ids],2)

    imf5s_train = np.expand_dims(imf5s[train_ids],2)
    imf5s_test = np.expand_dims(imf5s[test_ids],2)

    imf6s_train = np.expand_dims(imf6s[train_ids], 2)
    imf6s_test = np.expand_dims(imf6s[test_ids], 2)

    train_data = np.concatenate((ecgs_train, imf5s_train, imf6s_train), axis=2)
    test_data = np.concatenate((ecgs_test, imf5s_test, imf6s_test), axis=2)
    return train_data, test_data


class UWBECGDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        ecg_img = self.data[index, :, 0]
        imf5_img = self.data[index, :, 1]
        imf6_img = self.data[index, :, 2]
        ecg_img = torch.unsqueeze(torch.tensor(ecg_img), 0)
        imf5_img = torch.unsqueeze(torch.tensor(imf5_img), 0)
        imf6_img = torch.unsqueeze(torch.tensor(imf6_img), 0)

        return ecg_img, imf5_img, imf6_img

    def __len__(self):
        return self.data.shape[0]


def add_sine(data):
    t = np.arange(0, 10, 1/200)
    # amp = (max(data)-min(data))/8
    amp = (max(data) - min(data)) / 16
    frquency = random.randrange(15, 25, 1)/100

    sine_data = amp*np.sin(2*np.pi*frquency*t)
    new_data = data + sine_data
    new_data = (new_data - np.mean(new_data)) / np.std(new_data)
    new_data = mapminmax(new_data)
    return new_data


class UWBECG_Contra_DataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        ecg_img = self.data[index, :, 0]
        amplitude = self.data[index, :, 1]
        phase = self.data[index, :, 2]

        sine_amp = add_sine(amplitude)
        sine_pha = add_sine(phase)

        ecg_img = torch.unsqueeze(torch.tensor(ecg_img), 0)
        amplitude = torch.unsqueeze(torch.tensor(amplitude), 0)
        phase = torch.unsqueeze(torch.tensor(phase), 0)
        sine_amp = torch.unsqueeze(torch.tensor(sine_amp), 0)
        sine_pha = torch.unsqueeze(torch.tensor(sine_pha), 0)
        return ecg_img, amplitude, phase, sine_amp, sine_pha

    def __len__(self):
        return self.data.shape[0]
