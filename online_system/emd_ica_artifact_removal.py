from pprint import pprint
from typing import List, Tuple, Any, Union, Iterable

import PyEMD
import mne
import numpy as np
import pandas as pd
from EntropyHub import SampEn
from mne.io import RawArray, Raw
from mne.preprocessing import ICA
from numpy import ndarray
from pandas import DataFrame
from scipy.stats import kurtosis

from online_system.definitions import freq


def one_d_emd_decompose(data):
    eemd = PyEMD.CEEMDAN(trials=10)
    eemd.ceemdan(data)
    imfs, res = eemd.get_imfs_and_residue()
    return imfs


def get_imfs(data, ch_names):
    channel_number, data_length = data.shape
    pprint("artifact_removal, channel_number: " + str(channel_number) + " data_length: " + str(data_length))
    imf_data = None
    imf_ch_names = []
    for i in range(0, channel_number):
        channel_data = data[i]
        imfs = one_d_emd_decompose(channel_data)
        imf_number = imfs.shape[0]
        ch_name = ch_names[i]
        ch_name = ch_name.replace("EEG", "").strip()
        for i in range(0, imf_number):
            imf_ch_names.append(ch_name + " imf" + str(i + 1))
        if imf_data is None:
            imf_data = imfs
        else:
            imf_data = np.vstack((imf_data, imfs))
    return imf_data, imf_ch_names


def one_d_imf_sqa(imf, entropy_threshold=2, kurt_threshold=150) -> Tuple[Any, Union[ndarray, Iterable, int, float]]:
    data = np.squeeze(imf)
    (res, *_) = SampEn(data, m=3)
    entropy = res[0] - res[1]
    kurt = kurtosis(data)
    return entropy, kurt


def ica_component_sqa(ica_component, entropy_threshold=2.0, kurt_threshold=150) -> Tuple[float, float]:
    data = np.squeeze(ica_component)
    (res, *_) = SampEn(data, m=3)
    entropy = res[0] - res[1]
    kurt = kurtosis(data)
    return entropy, kurt


def imf_filtering(data, ch_names, entropy_threshold=2, kurt_threshold=100, no_filtering=False, logging=True) -> List[
    int]:
    res = DataFrame(data={'ch_name': [], 'entropy': [], 'kurt': []})
    channel_number = data.shape[0]
    for i in range(0, channel_number):
        entropy, kurt = one_d_imf_sqa(data[i], entropy_threshold, kurt_threshold)
        res = pd.concat([
            DataFrame(data={
                'ch_name': [ch_names[i]],
                'entropy': [entropy],
                'kurt': [kurt]
            }), res], sort=False, ignore_index=True
        )
    filtered_imf_info = res.query('entropy < {} and kurt < {}'.format(entropy_threshold, kurt_threshold))
    return filtered_imf_info.index


def ICA_decompose(data, input_ch_names) -> Tuple[Any, ICA, RawArray]:
    ch_names = input_ch_names if not isinstance(input_ch_names, np.ndarray) else list(input_ch_names)
    new_info = mne.create_info(ch_names, ch_types=["eeg"] * len(ch_names), sfreq=freq)
    raw = mne.io.RawArray(data, new_info)
    filt_raw = raw.copy().filter(l_freq=1., h_freq=None)
    n_components = 50 if len(ch_names) > 50 else int(0.8 * len(ch_names))
    ica = ICA(n_components=n_components, max_iter='auto', random_state=97, method='picard')
    ica.fit(filt_raw)
    ica_components = ica.get_sources(raw)
    # print(ica_components)
    return ica_components, ica, raw


def ica_components_filtering(ica_components_raw, ch_names, entropy_threshold=1.0, kurt_threshold=50, no_filtering=False, logging=True):
    data = ica_components_raw.get_data()
    res = DataFrame(data={'ch_name': [], 'entropy': [], 'kurt': []})
    channel_number = data.shape[0]
    for i in range(0, channel_number):
        entropy, kurt = ica_component_sqa(data[i], entropy_threshold, kurt_threshold)
        res = pd.concat([
            DataFrame(data={
                'ch_name': ['component ' + str(i+1)],
                'entropy': [entropy],
                'kurt': [kurt]
            }), res], sort=False, ignore_index=True
        )
    filtered_ica_component_info = res.query('entropy < {} and kurt < {}'.format(entropy_threshold, kurt_threshold))
    return filtered_ica_component_info.index


def ica_stage(data, input_ch_names) -> Raw:
    ica_components_raw, ica, data_raw = ICA_decompose(data, input_ch_names)
    picked_indexes = ica_components_filtering(ica_components_raw, input_ch_names)
    filtered_imfs = ica.apply(data_raw, include=picked_indexes.tolist())
    return filtered_imfs


def imfs_merge(imf_raw: Raw):
    imfs = imf_raw.get_data()
    imf_names = imf_raw.info['ch_names']
    df = DataFrame(columns=['imf_name'])
    for i in range(0, len(imf_names)):
        imf_name = imf_names[i].split(' ')[0]
        df = pd.concat((
                df,
                DataFrame(
                    data=[{
                            'imf_name': imf_name
                    }]
                )
        ),ignore_index=True)
    group_by = df.groupby('imf_name')
    groups = group_by.groups
    eeg_channels = None
    for key in groups.keys():
        indexes = groups[key]
        imfs_of_this_ch = imfs[indexes]
        eeg_channel = np.zeros(shape=(1, imfs.shape[1]))
        for i in range(0, len(indexes)):
            eeg_channel += np.array(imfs_of_this_ch[i])
        if eeg_channels is None:
            eeg_channels = eeg_channel
        else:
            eeg_channels = np.vstack((eeg_channels, eeg_channel))
    ch_names = list(groups.keys())
    new_info = mne.create_info(ch_names, ch_types=["eeg"] * len(ch_names), sfreq=freq)
    raw = mne.io.RawArray(eeg_channels, new_info)
    print(eeg_channels.shape)
    return raw

def artifact_removal(data, ch_names):
    get_imfs(data, ch_names)
