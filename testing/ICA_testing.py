import os

import mne
from mne.preprocessing import (ICA, create_eog_epochs)

from definitions import ROOT_DIR

record_path= os.path.join(ROOT_DIR, 'data', 'p10_Record1.edf')
raw = mne.io.read_raw_edf(record_path)
# raw.crop(0, 500).load_data()
raw.crop(0, 60).load_data()


# pick some channels that clearly show heartbeats and blinks
raw.plot(show_scrollbars=True)

eog_evoked = create_eog_epochs(raw, ch_name='EEG Fp2-Ref').average()
eog_evoked.apply_baseline(baseline=(None, -0.2))
# observe how much the channels react most to EOG
eog_evoked.plot()

# Filtering to remove slow drifts
filt_raw = raw.copy().filter(l_freq=1., h_freq=None)
ica = ICA(n_components=15, max_iter='auto', random_state=97)
ica.fit(filt_raw)
explained_var_ratio = ica.get_explained_variance_ratio(filt_raw)
for channel_type, ratio in explained_var_ratio.items():
    print(
        f'Fraction of {channel_type} variance explained by all components: '
        f'{ratio}'
    )