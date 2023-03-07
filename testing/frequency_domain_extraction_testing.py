from fearure.feature_extraction import frequency_domain_feature_extraction, feature_extraction_batching
from slicing import sliced_data

[normal_record_pieces, seizure_record_pieces] = record = sliced_data()[0]
signal_batch = seizure_record_pieces[0:2]

# res = frequency_domain_feature_extraction(signal)
# print(res)
res = feature_extraction_batching(signal_batch, 1, "test")

a = 1