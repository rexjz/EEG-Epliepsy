import os.path
from pprint import pprint

import mne
from EntropyHub import SampEn
from scipy.stats import kurtosis
import numpy as np

from definitions import DATA_ROOT

def remove_empty_dims():
    return 0


def imf_sqa(imf, label="imf"):
    data = np.squeeze(imf)
    (res, *_) = SampEn(data, m=3)
    entropy = res[0] - res[1]
    kurt = kurtosis(data)
    print(label + " " + " entropy: ", entropy, "kurt: ", kurt)
    return entropy, kurt


def ica_component_sqa():

    return 0


def imf_filtering(imfs, channel_names, entropy_threshold=2, kurt_threshold=2):
    for i in range(0, len(channel_names)):
        channel_name = channel_names[i]
        entropy, kurt = imf_sqa(imfs[i:i + 1], str(i) + ": " + channel_name)
        res.append({
            'ch_name': channel_name,
            'entropy': entropy,
            'kurt': kurt
        })
    pprint(res)


edf_path = os.path.join(DATA_ROOT, "imfs.edf")
raw = mne.io.read_raw_edf(edf_path, preload=True)
raw = raw.load_data()
raw.plot()
data = raw._data
# data = raw
res = []
channel_names = raw.info["ch_names"]
imf_filtering(data, channel_names)
