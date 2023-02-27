from pandas import Series
# Load NeuroKit and other useful packages
import neurokit2 as nk
import numpy as np
from scipy import signal
from sklearn.metrics import mean_squared_error
from generate_data import mapminmax
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import os
plt.rcParams['figure.figsize'] = [8, 5]  # Bigger images
import warnings
warnings.filterwarnings("ignore")


def cc(vector1, vector2):
    """
    calculate xcorr to estimate the shape of generated ecg
    :param extrated_ecg:
    :param ecg:
    :return:
    """

    vector1, vector2 = matlab_xcorr(vector1, vector2)

    vector1 = Series(vector1)
    vector2 = Series(vector2)

    corr = vector1.corr(vector2)
    return corr


def rmse(vector1, vector2):
    vector1 = vector1[:-100]
    vector2 = vector2[:-100]

    vector1, vector2 = matlab_xcorr(vector1, vector2)

    vector1 = mapminmax(vector1, 0, 1)
    vector1 = vector1 - np.mean(vector1)

    vector2 = mapminmax(vector2, 0, 1)
    vector2 = vector2 - np.mean(vector2)
    return np.sqrt(mean_squared_error(vector1, vector2))


def cos_dis(vector1, vector2):
    """
    estimate the cos distance of ground truth ecg and generated ecg
    :param vector1:
    :param vector2:
    :return:
    """
    vector1 = vector1[:-100]
    vector2 = vector2[:-100]

    vector1, vector2 = matlab_xcorr(vector1, vector2)

    dot_product = 0.0
    normA = 0.0
    normB = 0.0

    for a, b in zip(vector1, vector2):
        dot_product += a*b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return None
    else:
        return dot_product/((normA * normB) ** 0.5)


def matlab_xcorr(vector1, vector2):
    c = signal.correlate(vector1, vector2, 'full')
    lags = signal.correlation_lags(vector1.size, vector2.size, 'full')
    max_lag = lags[np.argmax(c)]
    if abs(max_lag) < lags[-1]:
        # if max lag <0, vector2 behind vector1
        if max_lag < 0:
            vector2 = vector2[abs(max_lag):]
            vector1 = vector1[:max_lag]
        elif max_lag > 0:
            vector1 = vector1[max_lag:]
            vector2 = vector2[:-max_lag]
    return vector1, vector2


def extractPQRST(ecg, gen_ecg, fs, save_dir, index, save_fig=True, show_fig=False):
    """
    从原始ECG或则UWB2ECG中提取peak。R、P、Q、S、T。
    :param ecg:
    :param gen_ecg:
    :param fs:
    :param save_dir:
    :param index:
    :param save_fig:
    :param show_fig:
    :return:
    """
    try:
        _, ecg_rpeaks = nk.ecg_peaks(ecg, sampling_rate=fs, method='nabian2018')
        _, ecg_waves_peak = nk.ecg_delineate(ecg, ecg_rpeaks, sampling_rate=fs, method="peak")

        _, gen_ecg_rpeaks = nk.ecg_peaks(gen_ecg, sampling_rate=fs, method='nabian2018')
        _, gen_ecg_waves_peak = nk.ecg_delineate(gen_ecg, gen_ecg_rpeaks, sampling_rate=fs, method="peak")

        if save_fig:
            nk.events_plot(ecg_rpeaks['ECG_R_Peaks'], ecg)
            plt.savefig(os.path.join(save_dir, str(index) + '_' + 'R_ecg' + '.png'))
            plt.close()
        if show_fig:
            nk.events_plot(ecg_rpeaks['ECG_R_Peaks'], ecg)
            plt.show()
            plt.close()

        if save_fig:
            nk.events_plot(gen_ecg_rpeaks['ECG_R_Peaks'], gen_ecg)
            plt.savefig(os.path.join(save_dir, str(index) + '_' + 'R_gen_ecg' + '.png'))
            plt.close()
        if show_fig:
            nk.events_plot(gen_ecg_rpeaks['ECG_R_Peaks'], gen_ecg)
            plt.show()
            plt.close()

        if save_fig:
            nk.events_plot([ecg_waves_peak['ECG_T_Peaks'],ecg_waves_peak['ECG_P_Peaks'],
                            ecg_waves_peak['ECG_Q_Peaks'],ecg_waves_peak['ECG_S_Peaks']], ecg)
            plt.savefig(os.path.join(save_dir, str(index) + '_' + 'TPQS_ecg' + '.png'))
            plt.close()
        if show_fig:
            nk.events_plot([ecg_waves_peak['ECG_T_Peaks'], ecg_waves_peak['ECG_P_Peaks'],
                            ecg_waves_peak['ECG_Q_Peaks'], ecg_waves_peak['ECG_S_Peaks']], ecg)
            plt.show()
            plt.close()

        if save_fig:
            nk.events_plot([gen_ecg_waves_peak['ECG_T_Peaks'],gen_ecg_waves_peak['ECG_P_Peaks'],
                            gen_ecg_waves_peak['ECG_Q_Peaks'],gen_ecg_waves_peak['ECG_S_Peaks']], gen_ecg)
            plt.savefig(os.path.join(save_dir, str(index) + '_' + 'TPQS_gen_ecg' + '.png'))
            plt.close()
        if show_fig:
            nk.events_plot([gen_ecg_waves_peak['ECG_T_Peaks'], gen_ecg_waves_peak['ECG_P_Peaks'],
                            gen_ecg_waves_peak['ECG_Q_Peaks'], gen_ecg_waves_peak['ECG_S_Peaks']], gen_ecg)
            plt.show()
            plt.close()

        # 计算间期
        # ecg间期 ground truth
        intervals_ecg = cal_interval(ecg_rpeaks, ecg_waves_peak, fs)
        intervals_gen_ecg = cal_interval(gen_ecg_rpeaks, gen_ecg_waves_peak, fs)
        # 如果心跳范围不在一分钟50-120次内，不返回
        if intervals_ecg is None or intervals_gen_ecg is None or \
                intervals_gen_ecg[0] > 60/40 or intervals_gen_ecg[0] < 60/120:
            # print('fail get intervals:', index)
            return None
        return intervals_ecg, intervals_gen_ecg
    except Exception as e:
        # print(index)
        # print(e)
        return None


def cal_interval(rpeaks, other_peak, fs):
    """
    从extractPQRST中提取的peak中去计算QRS间期、QT间期、PR间期、RR间期。
    :param rpeaks:
    :param other_peak:
    :param fs:
    :return:
    """
    R_indexs = np.asarray(rpeaks['ECG_R_Peaks'])
    T_indexs = np.asarray(other_peak['ECG_T_Offsets'])
    P_indexs = np.asarray(other_peak['ECG_P_Onsets'])

    # T_indexs = np.asarray(other_peak['ECG_T_Peaks'])
    # P_indexs = np.asarray(other_peak['ECG_P_Peaks'])

    Q_indexs = np.asarray(other_peak['ECG_Q_Peaks'])
    S_indexs = np.asarray(other_peak['ECG_S_Peaks'])
    last_R_index = 0

    QRS_list = []
    PR_list = []
    QT_list = []
    RR_list = []
    for R_index in R_indexs:
        # call RR interval 50-150次/分鐘
        temp = R_index - last_R_index
        if last_R_index != 0 and temp > 0.5 * fs and temp < fs * 1.5:
            RR_list.append(temp / fs)
        elif last_R_index != 0:
            last_R_index = R_index
            continue

        temp = np.where((Q_indexs > R_index-0.2*fs) & (Q_indexs < R_index))[0]
        if len(temp) > 0:
            Q_index = Q_indexs[temp[0]]
        else:
            Q_index = None
        temp = np.where((S_indexs > R_index) & (S_indexs < R_index+0.2*fs))[0]
        if len(temp) > 0:
            S_index = S_indexs[temp[0]]
        else:
            S_index = None
        temp = np.where((P_indexs > last_R_index) & (P_indexs < R_index))[0]
        if len(temp) > 0:
            P_index = P_indexs[temp[0]]
        else:
            P_index = None
        temp = np.where((T_indexs > R_index) & (T_indexs < min(R_index+fs, 10*fs)))[0]
        if len(temp) > 0:
            T_index = T_indexs[temp[0]]
        else:
            T_index = None
        # peak_pair
        if S_index and Q_index:
            QRS_list.append((S_index - Q_index)/fs)
        if P_index and Q_index:
            PR_list.append((Q_index - P_index)/fs)
        if T_index and Q_index:
            QT_list.append((T_index - Q_index)/fs)

        last_R_index = R_index

    if len(RR_list) > 5:
        RR_mediam = mediamsc(RR_list, 1)
        QRS_mediam = mediamsc(QRS_list, 1)
        PR_mediam = mediamsc(PR_list, 1)
        QT_mediam = mediamsc(QT_list, 1)
    else:
        return None
    return RR_mediam, QRS_mediam, PR_mediam, QT_mediam


def mediamsc(v, num):
    """
    Compute the median value of a vector excluding the distribution tails.
    :param v:
    :param perc:
    :return:
    """
    vo = np.sort(v)
    ii = num
    fi = len(v) - 1 - num
    x = np.nanmean(vo[ii:fi])
    return x


def plot_interval_bar(data, name, data_path):
    """
    plot bar of interval
    :param data:
    :param name:
    :param data_path:
    :return:
    """
    NaN_res = np.isnan(data)
    if True in NaN_res:
        pos = np.where(NaN_res == True)
        data = np.delete(data, pos)

    [nb, vp] = np.histogram(data)
    vp = vp[:-1] + (vp[1:] - vp[:-1]) / 2
    plt.bar(vp, nb, width=(vp[2] - vp[0]) / 4)
    plt.title(name)
    plt.savefig(os.path.join(data_path, name + '.png'))
    plt.close()


def internal_statistics(intervals, data_path):
    """
    plot distribution of all kinds of interval
    :param intervals:
    :param data_path:
    :return:
    """
    # 求不同间期的误差
    error = abs(intervals[:, 0:4] - intervals[:, 4:])

    median_err = np.nanmedian(error, 0)
    mean_err = np.nanmean(error, 0)
    # print('Median error')
    print('RR err:%.3f QRS err:%.3f PR err:%.3f QT err:%.3f' % (
    median_err[0], median_err[1], median_err[2], median_err[3]))
    return median_err
    # print('Mean error')
    # print('RR err:%.3f QRS err:%.3f PR err:%.3f QT err:%.3f' % (mean_err[0], mean_err[1], mean_err[2], mean_err[3]))


    # # 统计间期的分布
    # plot_interval_bar(intervals[:, 0], 'ECG_RR_interval', data_path)
    # plot_interval_bar(intervals[:, 1], 'ECG_QRS_interval', data_path)
    # plot_interval_bar(intervals[:, 2], 'ECG_PR_interval', data_path)
    # plot_interval_bar(intervals[:, 3], 'ECG_QT_interval', data_path)
    #
    # plot_interval_bar(intervals[:, 4], 'GEN_ECG_RR_interval', data_path)
    # plot_interval_bar(intervals[:, 5], 'GEN_ECG_QRS_interval', data_path)
    # plot_interval_bar(intervals[:, 6], 'GEN_ECG_PR_interval', data_path)
    # plot_interval_bar(intervals[:, 7], 'GEN_ECG_QT_interval', data_path)


def evaluate_ecg_interval(data_path):
    ecgs = np.load(data_path + 'ecgs.npy')
    gen_ecgs = np.load(data_path + 'gen_ecgs.npy')
    intervals = []  # RR_mediam, QRS_mediam, PR_mediam, QT_mediam
    for index in range(ecgs.shape[0]):
        # if index % 50 == 0:
            # print('evaluate process: %.2f%%' % (100*index/ecgs.shape[0]))
        ecg = ecgs[index, :]
        gen_ecg = gen_ecgs[index, :]
        os.makedirs(data_path + 'saved_pic', exist_ok=True)
        fs = 200
        interval = extractPQRST(ecg, gen_ecg, fs, data_path + 'saved_pic', index, save_fig=False, show_fig=False)
        if interval:
            intervals.append(list(interval[0])+list(interval[1]))

    median_err = internal_statistics(np.asarray(intervals), data_path + 'saved_pic')
    np.save(data_path + 'intervals.npy', np.asarray(intervals))
    return median_err


if __name__ == '__main__':
    """
    
    """
    data_path = '../images/pix2pix_uwb/test/npy/'
    # internal_statistics(np.load(data_path + 'intervals.npy'), data_path + 'saved_pic')
    evaluate_ecg_interval('../images/pix2pix_nature/test/npy/')