import argparse
from utils.datautil import *


def generate_uwb_data():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="uwbdata", help="nature_data or our_uwb or uwb200Hz")
    parser.add_argument("--train_rate", type=float, default=0.9, help="rate of train data")
    parser.add_argument("--sample_len", type=int, default=2000, help="length of signal")
    opt = parser.parse_args()

    ecgs, win_amplitude_datas, win_phase_datas = load_data(opt)
    train_data, test_data = trainTestSplit(ecgs, win_amplitude_datas, win_phase_datas, opt.train_rate)

    print('Train dataSet:', train_data.shape[0])
    print('test dataSet:', test_data.shape[0])

    np.save("data/train_uwb.npy", train_data)
    np.save("data/test_uwb.npy", test_data)
# Train dataSet: 14521
# test dataSet: 1614


def generate_nature_data():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="nature_data", help="nature_data or our_uwb")
    parser.add_argument("--train_rate", type=float, default=1, help="rate of train data")
    parser.add_argument("--sample_len", type=int, default=2000, help="length of signal")
    parser.add_argument("--except_file", type=str, default='GDN0011', help="length of signal")
    parser.add_argument("--only_file", type=str, default=None, help="length of signal")
    opt = parser.parse_args()

    ecgs, win_amplitude_datas, win_phase_datas = load_data(opt)
    train_data, test_data = trainTestSplit(ecgs, win_amplitude_datas, win_phase_datas, opt.train_rate)

    print('Train dataSet:', train_data.shape[0])
    print('test dataSet:', test_data.shape[0])

    np.save("data/nature_data_resting_for_train/train_nature_%s.npy" % (opt.except_file), train_data)
    # np.save("data/nature_data_resting_for_train/test_nature_%s.npy" % (opt.only_file), test_data)

    # np.save("data/train_nature.npy", train_data)
    # np.save("data/test_nature.npy", test_data)


if __name__ == '__main__':
    generate_nature_data()

