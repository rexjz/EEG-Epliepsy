import os
from definitions import ROOT_DIR
import pandas as pd
from constant import TIME_DOMAIN_FEATURE, FREQUENCY_DOMAIN_FEATURE

columns = list(set(TIME_DOMAIN_FEATURE + FREQUENCY_DOMAIN_FEATURE))

feature_home = os.path.join(ROOT_DIR, "data", "feature")


def flatten(df: pd.DataFrame):
    feature_names = None
    _df = df[columns]
    gk = _df.groupby('piece_index')
    res = []
    for (ground_id, group_df) in gk:
        group_df = group_df.drop(columns=['piece_index'])
        feature_names = group_df.columns
        res.append(group_df.to_numpy().flatten())
    return res, feature_names


# return [normal_features, seizure_features]
def read_feature():
    normal_features = []
    seizure_features = []
    g = os.walk(feature_home)
    feature_names = None
    for path, dir_list, file_list in g:
        for file_name in file_list:
            f = os.path.join(path, file_name)
            df = pd.read_csv(f)
            features, feature_names = flatten(df)
            if file_name.find('seizure') != -1:
                seizure_features.extend(features)
            elif file_name.find('normal') != -1:
                normal_features.extend(features)
    return normal_features, seizure_features, feature_names
