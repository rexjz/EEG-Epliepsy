import json
import logging
import os
import shutil
import sys
from datetime import datetime, date, time
from pathlib import Path

import mne
import numpy as np
import scipy.io
from concurrent.futures import ThreadPoolExecutor
from data_slicing.metadata import metadata
from definitions import EpilepticEEGDataset
from definitions import EpilepticEEGDataset_segments
from wica import wi

freq = 500  # 500Hz
STEP = 2 * freq  # 2 seconds


# convert_time_to_index(record_time, seizure_time)
# seizure_time = Tuple(hour, minute, second, seizure_duration)
def convert_time_to_index(record_time, seizure_time):
    (hour, minute, second, seizure_duration) = seizure_time
    s_time = time(hour, minute, second)
    diff = datetime.combine(date.today(), s_time) - record_time  # assigned date does not matter
    s_index = int(diff.total_seconds() * freq)  # 500 sampling rate,
    # s_index: seizure start in samples
    s_index_end = s_index + int(seizure_duration * freq)
    return s_index, s_index_end


def get_data(record) -> [np.array, np.array]:
    record_path = os.path.join(EpilepticEEGDataset, record["name"])
    if not os.path.exists(record_path):
        logging.error(record_path + " does not exit")
        return
    data = mne.io.read_raw_edf(record_path, preload=True, exclude=["Manual"])

    seizure_record = []
    index_mat = np.array([0, 0])
    raw_data = data.get_data()
    raw_data = np.array(raw_data)
    record_time = data.info['meas_date']
    record_time = record_time.time()
    record_time = datetime.combine(date.today(), record_time)
    print('=== record_time ===', record_time)

    for seizure_time in record["seizure_time"]:
        s_index, s_index_end = convert_time_to_index(record_time, seizure_time)
        print('s_index, s_index_end: ', s_index, s_index_end)
        st = raw_data[:, s_index:s_index_end]
        index_mat = np.vstack([index_mat, [s_index, s_index_end]])
        if len(seizure_record) == 0:
            seizure_record = st
        else:
            seizure_record = np.concatenate((seizure_record, st), axis=1)

    seizure_intervals = index_mat[1: len(index_mat) + 1]
    y = []
    for seizure_interval in seizure_intervals:
        x = np.linspace(seizure_interval[0], seizure_interval[1], seizure_interval[1] - seizure_interval[0],
                        endpoint=True)
        y.append(x)
    y = np.concatenate(y).astype(int)
    slice = np.s_[y]
    normal_record = np.delete(raw_data, slice, axis=1)

    # check normal data and seizure data
    print('=== raw_data.shape === ', raw_data.shape)
    print('=== normal_record.shape === ', normal_record.shape)
    print('=== seizure_record.shape === ', seizure_record.shape)
    return normal_record, seizure_record, data.info


def sliced_data(patient_code="p10"):
    patient_metadata = metadata[patient_code]
    records = patient_metadata["records"]
    res = {}
    for record in records:
        normal_record, seizure_record, info = get_data(record)
        # divide the data into pieces by every 2 seconds:
        normal_record_pieces = []
        seizure_record_pieces = []
        channel_number = normal_record.shape[0]
        for i in range(0, normal_record.shape[1], STEP):
            normal_piece = normal_record[:channel_number, i:i + STEP]
            normal_record_pieces.append(normal_piece)

        for i in range(0, seizure_record.shape[1], STEP):
            seizure_piece = seizure_record[:channel_number, i:i + STEP]
            seizure_record_pieces.append(seizure_piece)

        if patient_code not in res:
            res[patient_code] = {}

        if "record" not in res[patient_code]:
            res[patient_code]["record"] = []
            res[patient_code]["record_name"] = []

        res[patient_code]["record"].append((normal_record_pieces, seizure_record_pieces))
        res[patient_code]["record_name"].append(record["name"])
    return res[patient_code]["record"]


SEGMENT_LENGTH = 12800
SEGMENT_STORE = os.path.join(EpilepticEEGDataset, "segments")


def slice_and_save_data(patient_code="p10"):
    patient_store_path = os.path.join(SEGMENT_STORE, patient_code)
    if not os.path.exists(patient_store_path):
        os.makedirs(patient_store_path)

    patient_metadata = metadata[patient_code]
    records = patient_metadata["records"]
    for idx, record in enumerate(records):
        record_path = os.path.join(patient_store_path, "record" + str(idx + 1))
        if not os.path.exists(record_path):
            os.makedirs(record_path)
        info_path = os.path.join(record_path, "info.json")
        if os.path.exists(info_path):
            print("data exists")
            continue
        else:
            open(info_path, 'x')
        normal_record, seizure_record, info = get_data(record)
        with open(info_path, "w") as fp:
            json.dump({
                'ch_names': info['ch_names'],
                'sfreq': info['sfreq']
            }, fp)

        normal_record_path = os.path.join(record_path, "normal")
        if not os.path.exists(normal_record_path):
            os.makedirs(normal_record_path)
        counter = 0
        for idx in range(0, normal_record.shape[1], SEGMENT_LENGTH):
            if idx + SEGMENT_LENGTH > normal_record.shape[1]:
                continue
            segment = normal_record[:, idx: idx + SEGMENT_LENGTH]
            mat_path = os.path.join(normal_record_path, 'segment-' + str(counter) + '.mat')
            if not os.path.exists(mat_path):
                scipy.io.savemat(mat_path, {'data': segment})
                print('normal segment-' + str(counter) + '.mat saved')
            counter += 1

        seizure_record_path = os.path.join(record_path, "seizure")
        if not os.path.exists(seizure_record_path):
            os.makedirs(seizure_record_path)
        counter = 0
        for idx in range(0, seizure_record.shape[1], SEGMENT_LENGTH):
            if idx + SEGMENT_LENGTH > seizure_record.shape[1]:
                continue
            segment = seizure_record[:, idx: idx + SEGMENT_LENGTH]
            mat_path = os.path.join(seizure_record_path, 'segment-' + str(counter) + '.mat')
            scipy.io.savemat(mat_path, {'data': segment})
            print('seizure segment-' + str(counter) + '.mat saved')
            counter += 1


def get_segment_paths(patient_code):
    segments_folder = os.path.join(EpilepticEEGDataset_segments, patient_code)
    info = None
    normal_segment_paths = []
    seizure_segment_paths = []
    info = {}
    root = next(os.walk(segments_folder))[0]
    records = next(os.walk(segments_folder))[1]
    for record in records:
        normal_folder = os.path.join(segments_folder, os.path.join(record, 'normal'))
        seizure_folder = os.path.join(segments_folder, os.path.join(record, 'seizure'))
        info_path = os.path.join(segments_folder, os.path.join(record, 'info.json'))
        with open(info_path) as json_f:
            info = json.load(json_f)
        a = next(os.walk(normal_folder))
        normal_segment_paths.extend(
            list(map(lambda segment: os.path.join(normal_folder, segment), next(os.walk(normal_folder))[2]))
        )
        seizure_segment_paths.extend(
            list(map(lambda segment: os.path.join(seizure_folder, segment), next(os.walk(seizure_folder))[2]))
        )
    return normal_segment_paths, seizure_segment_paths, info


# slice_and_save_data()
# slice_and_save_data("p11")
# slice_and_save_data("p12")
# print("end")


def preprocess_segment(patient_code="p10"):
    normal_segment_paths, seizure_segment_paths, info = get_segment_paths(patient_code)
    ch_names = info['ch_names']
    sfreq = info['sfreq']
    paths = seizure_segment_paths.copy()
    paths.extend(normal_segment_paths)
    for segment_path in paths:
        try:
            print(segment_path)
            path = Path(segment_path.replace("segments", "preprocessed_segments"))
            if path.exists():
                continue
            if not os.path.exists(path.parent.absolute()):
                os.makedirs(path.parent.absolute())
            info_path = os.path.join(Path(segment_path).parent.parent.absolute(), 'info.json')
            new_info_path = Path(info_path.replace("segments", "preprocessed_segments"))
            if not new_info_path.exists():
                shutil.copy(info_path, new_info_path)
            mat_dict = scipy.io.loadmat(segment_path)
            data = mat_dict['data']
            res = wi.wi_for_data(data, ch_names, freq=sfreq)
            scipy.io.savemat(path.absolute(), {'data': res})
        except Exception as e:
            print("error", e)
    return 0


def do_preprocess_segment():
    if __name__ == '__main__':
        pool = ThreadPoolExecutor(max_workers=3)
        # results = list(pool.map(preprocess_segment, [
        #     ("p10"), ("p11"), ("p12")
        # ]))
        results = list(pool.map(preprocess_segment, [
            ("p11")
        ]))


do_preprocess_segment()
