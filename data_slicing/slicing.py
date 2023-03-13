import mne
import numpy as np
from datetime import datetime, date, time, timedelta
import data as da
from data_slicing.metadata import metadata
import os
import logging
from definitions import EpilepticEEGDataset

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
    data = mne.io.read_raw_edf(record_path, preload=True)
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
        if not seizure_record:
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
        normal_record, seizure_record = get_data(record)
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


def slice_and_save_data(patient_code="p10"):
    patient_metadata = metadata[patient_code]
    records = patient_metadata["records"]
    for record in records:
        normal_record, seizure_record = get_data(record)
