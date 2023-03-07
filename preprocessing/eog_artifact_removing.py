import mne
import os
from definitions import ROOT_DIR
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs, corrmap)
from mne.export import export_raw
filename = "p10_Record2"
record_path = os.path.join(ROOT_DIR, 'data', filename + '.edf')
raw = mne.io.read_raw_edf(record_path, preload=True)
# ICA is a computational intense operation
raw = raw.load_data().copy()
filt_raw = raw.copy().filter(l_freq=1., h_freq=None)
ica = ICA(n_components=15, method='picard', max_iter='auto', random_state=97)
ica.fit(filt_raw)
print("fit done!")
explained_var_ratio = ica.get_explained_variance_ratio(filt_raw)
for channel_type, ratio in explained_var_ratio.items():
    print(
        f'Fraction of {channel_type} variance explained by all components: '
        f'{ratio}'
    )
ica.apply(raw, exclude=[1, 2, 3, 5, 6, 7, 9, 10, 11, 14])
print("apply done!")
new_record_path = os.path.join(ROOT_DIR, 'data', filename+ '_ICA.edf')
export_raw(new_record_path, raw, fmt='edf', add_ch_type=True)
print("done!")