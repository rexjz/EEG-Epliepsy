import mne
import numpy as np
from sklearn.decomposition import FastICA

import wt


def wica_for_data(data: np.ndarray, ch_names, freq):
    ica = FastICA(n_components=len(ch_names),
                          random_state=97,
                          whiten='unit-variance')
    transposed = data.T
    ica_components = ica.fit(transposed).transform(transposed).T
    new_ica_components = []
    for ica_component in ica_components:
        new_ica_components.append(wt.wavelet_thresholding(
            ica_component
        ))
    res = ica.inverse_transform(np.array(new_ica_components).T).T
    return res


def wica_for_mne_raw(data: mne.io.RawArray):
    return wica_for_data(data.get_data(), data.info['ch_names'], data.info['sfreq'])
