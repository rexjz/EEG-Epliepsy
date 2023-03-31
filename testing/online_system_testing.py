from data_slicing.metadata import metadata
from data_slicing.slicing import get_data
from online_system.emd_ica_artifact_removal import get_imfs

patient_code = "p10"
patient_metadata = metadata[patient_code]
records = patient_metadata["records"]
record = records[0]
normal_record, seizure_record, info = get_data(record)


def get_imfs_it():
    test_data = normal_record[:, 0:2000]
    imfs, imf_ch_names = get_imfs(test_data, info["ch_names"])
    a = 1

get_imfs_it()