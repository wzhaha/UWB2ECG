import numpy as np
import matplotlib.pyplot as plt


def plot_interval_bar(data, name):
    NaN_res = np.isnan(data)
    if True in NaN_res:
        pos = np.where(NaN_res == True)
        data = np.delete(data, pos)

    [nb, vp] = np.histogram(data)
    vp = vp[:-1] + (vp[1:] - vp[:-1]) / 2
    plt.bar(vp, nb, width=(vp[2] - vp[0]) / 4)
    plt.title(name)
    plt.savefig('saved_pic/' + name + '.png')
    plt.close()


intervals = np.load('saved_pic/intervals.npy')
# 统计间期的分布
plot_interval_bar(intervals[:, 0], 'ECG_RR_interval')
plot_interval_bar(intervals[:, 1], 'ECG_QRS_interval')
plot_interval_bar(intervals[:, 2], 'ECG_PR_interval')
plot_interval_bar(intervals[:, 3], 'ECG_QT_interval')

plot_interval_bar(intervals[:, 4], 'GEN_ECG_RR_interval')
plot_interval_bar(intervals[:, 5], 'GEN_ECG_QRS_interval')
plot_interval_bar(intervals[:, 6], 'GEN_ECG_PR_interval')
plot_interval_bar(intervals[:, 7], 'GEN_ECG_QT_interval')

# 求不同间期的误差
error = abs(intervals[:, 0:4] - intervals[:, 4:])
median_err = np.nanmedian(error, 0)
mean_err = np.nanmean(error, 0)
print('Median error')
print('RR err:%.3f QRS err:%.3f PR err:%.3f QT err:%.3f' % (median_err[0], median_err[1], median_err[2], median_err[3]))
print('Mean error')
print('RR err:%.3f QRS err:%.3f PR err:%.3f QT err:%.3f' % (mean_err[0], mean_err[1], mean_err[2], mean_err[3]))

