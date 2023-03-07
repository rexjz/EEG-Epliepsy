import os
import numpy as np
import mne
from definitions import ROOT_DIR
import matplotlib.pyplot as plt

record_path= os.path.join(ROOT_DIR, 'data', 'EPI_odzysk', 'JANPRZ', 'JANPRZ_EEG_DATA.edf')

raw = mne.io.read_raw_edf(record_path)
# raw.crop(0, 500).load_data()
montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage, on_missing='warn')
raw.load_data()
print(raw.info['bads'])


def observe_electric_noise():
    plt.title("EEG PSD")
    figure = raw.compute_psd(tmax=np.inf, fmax=125).plot()
    # fig = raw.plot_psd(tmax=np.inf, fmax=125, average=True)
    # for ax in fig.axes[:2]:
    #     freqs = ax.lines[-1].get_xdata()
    #     psds = ax.lines[-1].get_ydata()
    #     for freq in (60, 120, 180, 240):
    #         idx = np.searchsorted(freqs, freq)
    #         ax.arrow(x=freqs[idx], y=psds[idx] + 18, dx=0, dy=-12, color='red',
    #                  width=0.1, head_width=3, length_includes_head=True)


def observe_eog_noise():
    eog_events = mne.preprocessing.find_eog_events(raw, ch_name='EEG Fp2-Ref')
    # 设置EOG事件的开始时间和持续时长（500ms）等
    onsets = eog_events[:, 0] / raw.info['sfreq'] - 0.25
    durations = [0.5] * len(eog_events)
    descriptions = ['bad blink'] * len(eog_events)
    blink_annot = mne.Annotations(onsets, durations, descriptions,
                                  orig_time=raw.info['meas_date'])
    raw.set_annotations(blink_annot)
    eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True)
    raw.plot()


def notch_filtering():
    raw.notch_filter(np.array([50]), fir_design='firwin')
    plt.title("EEG PSD")
    raw.compute_psd(tmax=np.inf, fmax=125).plot()
    raw.plot()


mne.set_config('MNE_BROWSE_RAW_SIZE', '3,3')
print(mne.get_config())
observe_electric_noise()
notch_filtering()
