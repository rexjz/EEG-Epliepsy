import numpy as np
import pywt


def stationary_wt(data, level, wavelet='coif5'):
    return pywt.swtn(data, wavelet=wavelet, level=6, start_level=0)


def get_threshold(data):
    np_data = np.array(data)
    noiselev = np.median(np.abs(data)) / 0.6745
    thresh = noiselev * np.sqrt(2 * np.log2(len(np_data)))
    return thresh


def soft_thresholding(data):
    np_data = np.array(data)
    thresh = get_threshold(np_data)
    print("thresh: " + str(thresh))
    gaps = np.abs(np_data) - thresh
    np_data[gaps < 0] = 0
    return np_data
