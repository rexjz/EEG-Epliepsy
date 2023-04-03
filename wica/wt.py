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


def wavelet_thresholding(data, wavelet='coif5'):
    coffs = stationary_wt(data, wavelet=wavelet, level=6)
    threshed_wt_coff = []
    for level_coff in coffs:
        threshed_wt_coff.append({
            'a': soft_thresholding(level_coff['a']),
            'd': soft_thresholding(level_coff['d'])
        })
    artifact_component = inverse_wt(threshed_wt_coff, wavelet=wavelet)
    artifact_free_data = np.array(data) - np.array(artifact_component)
    return artifact_free_data


def inverse_wt(coff, wavelet='coif5'):
    artifact_component = pywt.iswtn(coff, 'coif5')
    return artifact_component
