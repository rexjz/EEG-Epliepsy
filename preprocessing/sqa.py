import os.path

import mne
import numpy as np
import pandas
import pandas as pd
from EntropyHub import SampEn
from mne.preprocessing import ICA
from pandas import DataFrame
from scipy.stats import kurtosis

from definitions import DATA_ROOT


def imf_sqa(imf, label="imf"):
    data = np.squeeze(imf)
    (res, *_) = SampEn(data, m=3)
    entropy = res[0] - res[1]
    kurt = kurtosis(data)
    print(label + " " + " entropy: ", entropy, "kurt: ", kurt)
    return entropy, kurt


def ica_component_sqa(ica_component, label="ica_component"):
    data = np.squeeze(ica_component)
    (res, *_) = SampEn(data, m=3)
    entropy = res[0] - res[1]
    kurt = kurtosis(data)
    print(label + " " + " entropy: ", entropy, "kurt: ", kurt)
    return entropy, kurt


def ica_components_filtering(ica_components, component_names, entropy_threshold=2, kurt_threshold=50):
    res = DataFrame(data={'ch_name': [], 'entropy': [], 'kurt': []})
    n_components = ica_components.shape[0]
    for i in range(0, n_components):
        component_name = component_names[i]
        ica_component = ica_components[i]
        entropy, kurt = ica_component_sqa(ica_component)
        res = pd.concat([
            DataFrame(data={
                'component_name': [component_name],
                'entropy': [entropy],
                'kurt': [kurt]
            }), res], sort=False, ignore_index=True
        )
    return res


def imf_filtering(imfs, channel_names, entropy_threshold=2, kurt_threshold=150, no_filtering=False):
    res = DataFrame(data={'ch_name': [], 'entropy': [], 'kurt': []})
    for i in range(0, len(channel_names)):
        channel_name = channel_names[i]
        entropy, kurt = imf_sqa(imfs[i:i + 1], str(i) + ": " + channel_name)
        if no_filtering or (entropy < entropy_threshold and kurt < kurt_threshold):
            res = pd.concat([
                DataFrame(data={
                    'ch_name': [channel_name],
                    'entropy': [entropy],
                    'kurt': [kurt]
                }), res], sort=False, ignore_index=True
            )
    return res


def read_imfs():
    edf_path = os.path.join(DATA_ROOT, "imfs.edf")
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    raw = raw.load_data()
    raw.plot()
    data = raw._data
    channel_names = raw.info["ch_names"]
    return data, channel_names, raw


def test():
    data, channel_names = read_imfs()
    res = imf_filtering(data, channel_names)
    res.to_csv("imf_sqa_results.csv")
    print("finished")


def testICAFiltering():
    data, channel_names, raw = read_imfs()
    df = pandas.read_csv("imf_sqa_full_results.csv")
    filtered_imf_info = df.query('entropy < 2 and kurt < 150')
    filtered_ch_names = filtered_imf_info.loc[:, "ch_name"]
    raw = raw.pick_channels(np.array(filtered_ch_names))
    filt_raw = raw.copy().filter(l_freq=1., h_freq=None)
    n_components = len(filtered_ch_names)
    components_cap = 50
    if n_components > components_cap:
        n_components = components_cap
    ica = ICA(n_components=n_components, method='picard', max_iter='auto', random_state=97)
    ica.fit(filt_raw)
    remaining_sources = n_components
    for i in range(0, n_components, 20):
        if remaining_sources < 20:
            ica.plot_sources(raw, picks=range(i, i + remaining_sources, 1), show_scrollbars=False, stop=10)
        else:
            ica.plot_sources(raw, picks=range(i, i + 20, 1), show_scrollbars=False, stop=10)
        remaining_sources -= 20
    ica_components = ica.get_sources(raw)
    res = ica_components_filtering(ica_components._data, ica_components.info["ch_names"])
    included_ica = res.query('entropy < 1.5 and kurt < 75')
    ica_picked_indexes = np.array(included_ica.index).tolist()
    ica_after_picking = raw.copy()
    ica.apply(ica_after_picking, include=ica_picked_indexes)
    mne.export.export_raw("imfs_after_ICA_filtering3.edf", ica_after_picking, overwrite=True)
    print("finish")

testICAFiltering()
