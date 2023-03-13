import numpy as np
import pywt.data
from data_slicing.slicing import sliced_data

[normal_record_pieces, seizure_record_pieces] = record = sliced_data()[0]
signal = normal_record_pieces[0]
channel_number = signal.shape[0]
wp = pywt.WaveletPacket(data=signal, wavelet='db1', mode='symmetric', maxlevel=6)
nodes = wp.get_level(6, order="freq")
leave_number = len(nodes)
leave_data_length = nodes[0].data.shape[1]
E_i = np.zeros([leave_number, channel_number, leave_data_length])
for idx, node in enumerate(nodes):
    E_i[idx] = 0
    E_i[idx] = np.power(node.data, 2)

s = np.sum(E_i, axis=2)
E_total = np.sum(E_i, axis=2).reshape([leave_number, channel_number, 1])
q_i = E_i / E_total
entropy = np.sum(np.log(q_i) * q_i * -1, axis=2)
a = 10