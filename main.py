from data_slicing.slicing import *
from fearure.feature_extraction import feature_extraction_batching
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from definitions import EpilepticEEGDataset_artifact_free_segments

p_code = "p10"
if len(sys.argv) == 2:
    p_code = sys.argv[1]
    print("p_code: ", p_code)


def get_all_artifact_free_segments_asb_path(patient_code):
    segments_folder = os.path.join(EpilepticEEGDataset_artifact_free_segments, patient_code)
    normal_segment_paths = []
    seizure_segment_paths = []
    root = next(os.walk(segments_folder))[0]
    records = next(os.walk(segments_folder))[1]
    for record in records:
        normal_folder = os.path.join(segments_folder, os.path.join(record, 'normal'))
        seizure_folder = os.path.join(segments_folder, os.path.join(record, 'seizure'))
        a = next(os.walk(normal_folder))
        normal_segment_paths.extend(
            list(map(lambda segment: os.path.join(normal_folder, segment), next(os.walk(normal_folder))[2]))
        )
        seizure_segment_paths.extend(
            list(map(lambda segment: os.path.join(seizure_folder, segment), next(os.walk(seizure_folder))[2]))
        )
    return normal_segment_paths, seizure_segment_paths


piece_length = 2000


def feature_extraction_process(*, batch_size=100, max_workers=3):
    seizure_record_pieces = []
    normal_record_pieces = []
    normal_segment_paths10, seizure_segment_paths10 = get_all_artifact_free_segments_asb_path(p_code)
    seizure_segment_paths = []
    normal_segment_paths = []
    seizure_segment_paths.extend(seizure_segment_paths10)
    normal_segment_paths.extend(normal_segment_paths10)
    # seizure_segment_paths.extend(normal_segment_paths10)

    for p in seizure_segment_paths:
        data = mne.io.read_raw_edf(p, preload=True)
        raw_data = data.get_data()
        raw_data = np.array(raw_data)
        for index in range(0, raw_data.shape[1], piece_length):
            seizure_record_pieces.append(raw_data[:, index: index + piece_length])

    for p in normal_segment_paths:
        data = mne.io.read_raw_edf(p, preload=True)
        raw_data = data.get_data()
        raw_data = np.array(raw_data)
        for index in range(0, raw_data.shape[1], piece_length):
            normal_record_pieces.append(raw_data[:, index: index + piece_length])

    pieces_count = len(normal_record_pieces) + len(seizure_record_pieces)
    print("pieces_count: " + str(pieces_count))
    progressBar = tqdm.tqdm(
        total=pieces_count,
        desc="feature extraction",
    )
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for record_type, record in enumerate([seizure_record_pieces, normal_record_pieces]):
            if record_type == 0:
                record_type = 1
            else:
                record_type = 0
            record_len = len(record)
            batch_index = 0
            data_batch = []
            for index, piece in enumerate(record):
                data_batch.append(piece)
                if len(data_batch) >= batch_size or index == record_len - 1:
                    futures.append(
                        ex.submit(feature_extraction_batching, data_batch.copy(), record_type,
                                  "batch" + str(batch_index))
                    )
                    print("type" + str(record_type) + " batch" + str(batch_index) + " submitted")
                    batch_index += 1
                    data_batch.clear()
        for future in as_completed(futures):
            result = future.result()
            progressBar.update(batch_size)
    progressBar.close()


def test():
    seizure_record_pieces = []
    normal_record_pieces = []
    normal_segment_paths10, seizure_segment_paths10 = get_all_artifact_free_segments_asb_path("p10")
    seizure_segment_paths = []
    normal_segment_paths = []
    seizure_segment_paths.extend(seizure_segment_paths10)
    normal_segment_paths.extend(normal_segment_paths10)
    # seizure_segment_paths.extend(normal_segment_paths10)

    for p in seizure_segment_paths:
        data = mne.io.read_raw_edf(p, preload=True)
        raw_data = data.get_data()
        raw_data = np.array(raw_data)
        for index in range(0, raw_data.shape[1], piece_length):
            seizure_record_pieces.append(raw_data[:, index: index + piece_length])

    for p in normal_segment_paths:
        data = mne.io.read_raw_edf(p, preload=True)
        raw_data = data.get_data()
        raw_data = np.array(raw_data)
        for index in range(0, raw_data.shape[1], piece_length):
            normal_record_pieces.append(raw_data[:, index: index + piece_length])

    a = 0

# test()
feature_extraction_process(batch_size=10)
