import math
import os.path
from pprint import pprint

import numpy as np
from EntropyHub import FuzzEn
from scipy.stats import kurtosis
from definitions import DATA_ROOT
import mne


def imf_sqa(imf, label="imf"):
    # pdf = norm.pdf(imf)
    # entropy = norm.entropy(imf)
    (res, *_) = FuzzEn(imf, r=(0.15, (0.1 + np.std(imf)) / 2), m=3)
    entropy = res[0] - res[1]
    kurt = kurtosis(imf)
    print(channel_name + " entropy: ", entropy, "kurt: ", kurt)
    return entropy, kurt


edf_path = os.path.join(DATA_ROOT, "imfs.edf")
raw = mne.io.read_raw_edf(edf_path, preload=True)
raw = raw.load_data()
raw.plot()
data = raw._data
# data = raw
res = []
channel_names = raw.info["ch_names"]
for i in range(0, len(channel_names)):
    channel_name = channel_names[i]
    entropy, kurt = imf_sqa(data[i:i+1, :][0], channel_name)
    res.append({
        'ch_name': channel_name,
        'entropy': entropy,
        'kurt': kurt
    })

pprint(res)