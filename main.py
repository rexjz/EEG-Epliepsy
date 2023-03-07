from slicing import *
from fearure.feature_extraction import feature_extraction_batching
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def feature_extraction_process(*, batch_size=100, max_workers=8):
    records = sliced_data()
    normal_record_pieces = []
    seizure_record_pieces = []
    for (partial_normal_record_pieces, partial_seizure_record_pieces) in records:
        normal_record_pieces.extend(partial_normal_record_pieces)
        seizure_record_pieces.extend(partial_seizure_record_pieces)
    a = seizure_record_pieces[0]
    # time domain feature extraction:
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
                        ex.submit(feature_extraction_batching, data_batch.copy(), record_type, "batch"+str(batch_index))
                    )
                    print("type" + str(record_type) + " batch" + str(batch_index) + " submitted")
                    batch_index += 1
                    data_batch.clear()
        for future in as_completed(futures):
            result = future.result()
            progressBar.update(batch_size)
    progressBar.close()


feature_extraction_process(batch_size=10)
