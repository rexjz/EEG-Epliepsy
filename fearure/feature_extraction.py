import os.path
import sys
from pprint import pprint

import numpy as np
from scipy.stats import skew, kurtosis
import math
from EntropyHub._FuzzEn import FuzzEn
from EntropyHub import SampEn
from data_slicing.slicing import STEP
from pandas import DataFrame
import pandas as pd
from constant import TIME_DOMAIN_FEATURE, FREQUENCY_DOMAIN_FEATURE
from definitions import ROOT_DIR
from scipy.fftpack import fft
from constant import THETA_FREQ_RANGE, DELTA_FREQ_RANGE, ALPHA_FREQ_RANGE, BETA_FREQ_RANGE, GAMMA_FREQ_RANGE, \
    SAMPLING_FREQUENCY
import pywt
import yasa
from scipy import signal as sci_signal

feature_home = os.path.join(ROOT_DIR, 'data', 'feature')
if not os.path.exists(feature_home):
    os.makedirs(feature_home)


# batch: iterateable of pieces
# datatype:
#   0: normal record
#   1: seizure record
def feature_extraction_batching(batch, datatype, info):
    feature_table = DataFrame(dtype='float32')
    for piece_index, piece in enumerate(batch):
        new_feature = {}
        new_feature.update(time_domain_feature_extraction(piece, piece_index))
        new_feature.update(frequency_domain_feature_extraction(piece, piece_index))
        new_df = DataFrame(new_feature)
        bp = band_power_feature_extraction(piece)
        new_df = new_df.join(bp, how='outer')
        new_df = new_df.join(wp_entropy(piece), how='outer')
        feature_table = pd.concat([feature_table, new_df], ignore_index=True)
    if datatype == 0:
        p = os.path.join(feature_home, info + "-features-normal.csv")
        # save feature_table as normal feature
        feature_table.to_csv(p)
    else:
        p = os.path.join(feature_home, info + "-features-seizure.csv")
        # save feature_table as seizure feature
        feature_table.to_csv(p)
    return feature_table


# return channel_number, signal_length
def get_signal_info(signal):
    N = 0
    channel_number = 0
    if signal.ndim == 1:
        channel_number = 1
        N = len(signal)
    elif signal.ndim == 2:
        channel_number = signal.shape[0]
        N = signal.shape[1]
    return channel_number, N


def time_domain_feature_extraction(signal, piece_index=0):
    channel_number, _ = get_signal_info(signal)
    piece_index_column = np.full([channel_number, ], piece_index)
    mean = np.average(signal, axis=1)
    var = np.var(signal, axis=1)
    std = np.std(signal, axis=1)
    rms = np.zeros([channel_number, ])
    fuzzy_en = np.zeros([channel_number, ])
    sample_en = np.zeros([channel_number, ])
    skew_value = skew(signal, axis=1)
    kurt = kurtosis(signal, axis=1)
    abs_max = np.amax(np.abs(signal), axis=1)
    for channel_idx in range(0, channel_number):
        current_channel_data = signal[channel_idx:channel_idx + 1, :STEP]
        rms[channel_idx] = math.sqrt(
            np.sum(np.power(current_channel_data, 2)) / STEP
        )
        tolerance = np.std(current_channel_data) * 0.15
        (res, *_) = FuzzEn(current_channel_data, r=(tolerance, (0.1 + std[channel_idx]) / 2), m=3)
        fuzzy_en[channel_idx] = res[0] - res[1]
        (res, *_) = SampEn(current_channel_data, m=3, r=tolerance)
        sample_en[channel_idx] = res[0] - res[1]
    # TODO: 归一化
    feature_row = {
        'mean': mean,
        'var': var,
        'std': std,
        'rms': rms,
        'skew': skew_value,
        'kurt': kurt,
        'abs_max': abs_max,
        'fuzzy_en': fuzzy_en,
        'sample_en': sample_en,
        'piece_index': piece_index_column
    }
    return feature_row


def frequency_domain_feature_extraction(signal, piece_index=0):
    channel_number, N = get_signal_info(signal)
    piece_index_column = np.full([channel_number, ], piece_index)
    features = {'piece_index': piece_index_column}
    # features.update(wavelet_feature_extraction(signal, channel_number, N))
    return features


def band_power_feature_extraction(signal: np.ndarray, sf=500):
    if (signal.ndim == 1):
        len = signal.shape[0]
    else:
        len = signal.shape[1]
    win = 4 * sf
    freqs, psd = sci_signal.welch(signal, sf, nperseg=win)
    res = yasa.bandpower_from_psd(psd, freqs)
    r_res = res.iloc[:, 1:8]
    r_res.columns = [
        'delta_power_ratio', 'theta_power_ratio',
        'alpha_power_ratio', 'sigma_power_ratio',
        'beta_power_ratio', 'gamma_power_ratio',
        'total_power'
    ]
    return r_res


def fft_feature_extraction(signal, channel_number, N):
    theta_power_ratio = np.zeros([channel_number, ])
    delta_power_ratio = np.zeros([channel_number, ])
    alpha_power_ratio = np.zeros([channel_number, ])
    beta_power_ratio = np.zeros([channel_number, ])
    gamma_power_ratio = np.zeros([channel_number, ])
    power_ratios = [theta_power_ratio, delta_power_ratio, alpha_power_ratio, beta_power_ratio, gamma_power_ratio]
    amplitude_spectrum = (np.abs(fft(signal)) / N)[0: int(N / 2)]
    theta_range = ftt_freq_range2index_slice(THETA_FREQ_RANGE, N)
    delta_range = ftt_freq_range2index_slice(DELTA_FREQ_RANGE, N)
    alpha_range = ftt_freq_range2index_slice(ALPHA_FREQ_RANGE, N)
    beta_range = ftt_freq_range2index_slice(BETA_FREQ_RANGE, N)
    gamma_range = ftt_freq_range2index_slice(GAMMA_FREQ_RANGE(N / 2), N)
    for channel_idx in range(0, channel_number):
        total_power = np.sum(amplitude_spectrum[channel_idx])
        for index, slice_obj in enumerate([theta_range, delta_range, alpha_range, beta_range, gamma_range]):
            section = amplitude_spectrum[channel_idx][slice_obj]
            power_ratios[index][channel_idx] = np.sum(section) / total_power
    return {
        'theta_power_ratio': theta_power_ratio,
        'delta_power_ratio': delta_power_ratio,
        'alpha_power_ratio': alpha_power_ratio,
        'beta_power_ratio': beta_power_ratio,
        'gamma_power_ratio': gamma_power_ratio,
    }


def ftt_freq_range2index_slice(freq_range, n):
    return np.s_[ftt_freq2index(freq_range[0], n): ftt_freq2index(freq_range[1], n)]


def ftt_freq2index(freq, n):
    # index * fs / N = freq <==> freq * N / fs = index
    return int(freq * n / SAMPLING_FREQUENCY)


def compute_shannon_entropy(signal):
    """Here, we will compute nonnormalized shannon entropy.
    We will basically implement MATLAB function 'wentropy'.
    Input: signal should be 1D numpy array."""
    return -np.nansum(signal ** 2 * np.log(signal ** 2))  # nansum to exclude pesky terms like "0*np.log(0)"


def lowpassFiltering(data, cutoff_freq=35, fs=500):
    sos = sci_signal.butter(4, cutoff_freq, btype='lowpass', output='sos', fs=fs)
    return sci_signal.sosfilt(sos, data)


def wp_entropy(signal: np.ndarray, maxlevel=3, wavelet='db5'):
    filtedData = lowpassFiltering(signal)
    feature_matrix_wav_packet_entropy = np.repeat(np.nan, filtedData.shape[0] * 2 ** maxlevel) \
        .reshape(filtedData.shape[0], 2 ** maxlevel)
    for i in range(filtedData.shape[0]):
        wp = pywt.WaveletPacket(filtedData[i, :], wavelet=wavelet, maxlevel=maxlevel)  # Wavelet packet transformation
        packet_names = [node.path for node in wp.get_level(maxlevel, "natural")]  # Packet node names.
        for j in range(len(packet_names)):
            new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet, maxlevel=maxlevel)
            new_wp[packet_names[j]] = wp[packet_names[j]].data
            reconstructed_signal = new_wp.reconstruct(
                update=False)  # Signal reconstruction from wavelet packet coefficients
            feature_matrix_wav_packet_entropy[i, j] = compute_shannon_entropy(
                reconstructed_signal)  # Entropy of reconstructed signal for every node
    wp_entropy = {}
    for column in range(0, feature_matrix_wav_packet_entropy.shape[1]):
        wp_entropy['wp_entropy_' + str(column)] = feature_matrix_wav_packet_entropy[:, column]
    return DataFrame(wp_entropy)


def wavelet_feature_extraction(signal, channel_number, N):
    wp = pywt.WaveletPacket(data=signal, wavelet='db5', mode='symmetric', maxlevel=6)
    nodes = wp.get_level(6, order="freq")
    leave_number = len(nodes)
    leave_data_length = nodes[0].data.shape[1]
    E_i = np.zeros([leave_number, channel_number])
    for idx, node in enumerate(nodes):
        E_i[idx] = 0
        E_i[idx] = np.sum(np.power(node.data, 2), axis=1)

    E_i = E_i.reshape([E_i.shape[1], E_i.shape[0]])
    E_total = np.sum(E_i, axis=1).reshape(E_i.shape[0], 1)
    q_i = E_i / E_total
    entropy = np.sum(np.log(q_i) * q_i * -1, axis=1)
    features = {
        "shannon_wavelet_entropy": entropy
    }
    return features

# 1. SVM
# 2. one-d CNN 97%
# 3.
