import PyEMD
import mne
import numpy as np
from PyEMD import EMD, Visualisation, CEEMDAN
from mne.preprocessing import ICA

from data_slicing.metadata import metadata
from data_slicing.slicing import get_data, freq


def polt_imfs(imfs, res, t):
    vis = Visualisation()
    vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)
    vis.show()


def testEMD(patient_code="p10"):
    global normal_record
    patient_metadata = metadata[patient_code]
    records = patient_metadata["records"]
    record = records[0]
    normal_record, seizure_record, info = get_data(record)
    used_reacord = normal_record
    channel_number = used_reacord.shape[0]
    ch_names = info["ch_names"]
    imf_from_all_channels = []
    hyper_ch_names = []
    start = 1000
    length = 8000
    for i in range(0, channel_number):
        t = np.linspace(0, length / freq, length)
        S = used_reacord[i][start:start + length]
        # emd = EMD()
        # emd.emd(S)
        # eemd = PyEMD.EEMD(trials=10)
        eemd = PyEMD.CEEMDAN(trials=10)
        eemd.ceemdan(S)
        imfs, res = eemd.get_imfs_and_residue()
        polt_imfs(imfs, res, t)
        eeg_ch_name = ch_names[i].replace("EEG", "").strip()
        for imf_counter in range(0, len(imfs)):
            imf = imfs[imf_counter]
            hyper_ch_name = eeg_ch_name + "+" + "imf" + str(imf_counter + 1)
            hyper_ch_names.append(hyper_ch_name)
            imf_from_all_channels.append(imf)
    new_info = mne.create_info(hyper_ch_names, ch_types=["eeg"] * len(hyper_ch_names), sfreq=freq)
    # info.set_montage('standard_1020')
    simulated_raw = mne.io.RawArray(imf_from_all_channels, new_info)
    simulated_raw.plot(show_scrollbars=False, show_scalebars=False, title="simulated_raw")
    original_raw = mne.io.RawArray(used_reacord[:, start: start + length], info)
    # original_raw.plot(show_scrollbars=False, show_scalebars=False, title="original_raw")
    mne.export.export_raw("imfs.edf", simulated_raw, overwrite=True)


def emd_ica():
    simulated_raw = mne.io.read_raw_edf("imfs.edf", preload=True)
    hyper_ch_names = simulated_raw.info["ch_names"]
    filt_raw = simulated_raw.copy().filter(l_freq=1., h_freq=None)
    n_components = len(hyper_ch_names)
    components_cap = 100
    if n_components > components_cap:
        n_components = components_cap
    ica = ICA(n_components=n_components, method='picard', max_iter='auto', random_state=97)
    ica.fit(filt_raw)
    remaining_sources = n_components
    for i in range(0, n_components, 20):
        if remaining_sources < 20:
            ica.plot_sources(simulated_raw, picks=range(i, i + remaining_sources, 1), show_scrollbars=False, stop=10)
        else:
            ica.plot_sources(simulated_raw, picks=range(i, i + 20, 1), show_scrollbars=False, stop=10)
        remaining_sources -= 20
    print("finish")


# testEMD()
emd_ica()