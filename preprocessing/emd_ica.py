import json
import os.path
import shutil
from pathlib import Path
from pprint import pprint

import numpy as np
import scipy.io

from online_system import emd_ica_artifact_removal


def emd_ica(data, ch_names, freq, segment_path=''):
    imfs, imf_ch_names = emd_ica_artifact_removal.get_imfs(data, ch_names)
    print("imf_ch_names")
    pprint(imf_ch_names)
    eligible_imf_indexes = emd_ica_artifact_removal.imf_filtering(imfs, imf_ch_names)
    print("eligible_imf_indexes, select " + str(len(eligible_imf_indexes)) + " out of " + str(len(imf_ch_names)))
    pprint(eligible_imf_indexes)
    eligible_imfs = imfs[eligible_imf_indexes]
    eligible_imf_ch_names = np.array(imf_ch_names)[eligible_imf_indexes]
    imfs_raw = emd_ica_artifact_removal.ica_stage(eligible_imfs, eligible_imf_ch_names)
    raw_without_artifacts = emd_ica_artifact_removal.imfs_merge(imfs_raw)
    try:
        path = Path(segment_path.replace("preprocessed_segments", "good_segments"))
        if not path.parent.exists():
            os.makedirs(path.parent)
        info_path = os.path.join(Path(segment_path).parent.parent.absolute(), 'info.json')
        new_info_path = Path(info_path.replace("preprocessed_segments", "good_segments"))
        if not new_info_path.exists():
            shutil.copy(info_path, new_info_path)
        scipy.io.savemat(path, {'data': np.array(raw_without_artifacts.get_data())})
    except Exception as e:
        print(e)
    return raw_without_artifacts


def emd_ica_by_mat_path(segment_path):
    segment_path = Path(segment_path)
    info_path = os.path.join(segment_path.parent.parent.absolute(), 'info.json')
    info = {}
    with open(info_path) as jsonfile:
        info = json.load(jsonfile)
    data = scipy.io.loadmat(str(segment_path.absolute()))['data']
    ch_names = info["ch_names"]
    sfreq = info["sfreq"]
    return emd_ica(data, ch_names, sfreq, str(segment_path.absolute()))
