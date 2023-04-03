import signal

import mne
import numpy as np
from sklearn.decomposition import FastICA
from scipy import signal
from wica.wt import wavelet_thresholding


def wi_for_data(input_data: np.ndarray, ch_names, freq):
    sos = signal.butter(8, (1), btype='highpass', output='sos', fs=freq)
    data = signal.sosfilt(sos, input_data)
    ica = FastICA(n_components=len(ch_names),
                  random_state=97,
                  max_iter=400,
                  tol=1e-4,
                  whiten='unit-variance')
    transposed = data.T
    ica_components = ica.fit(transposed).transform(transposed).T
    new_ica_components = []
    for ica_component in ica_components:
        new_ica_components.append(wavelet_thresholding(
            ica_component
        ))
    res = ica.inverse_transform(np.array(new_ica_components).T).T
    return res


def wica_for_mne_raw(data: mne.io.RawArray):
    return wi_for_data(data.get_data(), data.info['ch_names'], data.info['sfreq'])
