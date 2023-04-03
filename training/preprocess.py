from pprint import pprint

import numpy as np
from sklearn.preprocessing import scale
from sklearn.feature_selection import VarianceThreshold
from read_features import read_feature
import numpy.ma as ma

channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'A1', 'A2']


def pre_process():
    normal_features, seizure_features, feature_names = read_feature()
    normal_features = np.array(normal_features)
    seizure_features = np.array(seizure_features)
    # 0-normal 1-seizure
    targets = np.hstack([np.zeros(len(normal_features)), np.ones(len(seizure_features))])
    features = np.concatenate([normal_features, seizure_features])

    #  variance threshold feature selection
    raw = {
        'features': features,
    }
    vt = VarianceThreshold(threshold=.3)
    features = vt.fit_transform(features)
    masks = vt.get_support()
    raw['masks'] = masks
    a = 10
    features = scale(features)  # z-score norm
    return { 'data': features, 'targets': targets, 'raw': raw, 'feature_names': feature_names }


def observe_feature_selection():
    data_set = pre_process()
    old_masks = data_set['raw']['masks']
    masks = np.zeros((len(old_masks), ))
    for i in range(0, len(old_masks)):
        masks[i] = int(old_masks[i])
    features = data_set['raw']['features']
    feature_names = data_set['feature_names']
    flatten_feature_names = []
    for i in range (0, 19):
        channel_name = channels[i] + ' '
        for feature_name in feature_names:
            flatten_feature_names.append(channel_name + feature_name)
    mx = ma.masked_array(flatten_feature_names, mask=masks)
    filtered = []
    for idx, mask in enumerate(old_masks):
        if mask:
            filtered.append(flatten_feature_names[idx])
    pprint(filtered)


observe_feature_selection()