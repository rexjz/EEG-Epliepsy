import numpy as np
from scipy.fftpack import fft, ifft
from slicing import *
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['KaiTi']  # 避免中文title乱码

[normal_record_pieces, seizure_record_pieces] = record = sliced_data()[0]
y = normal_record_pieces[0]
N = 0
if y.ndim == 1:
    N = len(y)
else:
    N = y.shape[1]

x = np.linspace(0, N, num=N, endpoint=False).astype(int)
fft_y = fft(y)
print(type(y[1]))
abs_y = np.abs(fft_y) / N  # 归一化振幅谱

x = list(map(lambda x_value: x_value * 500 / N, x[0: int(N/2)]))
for idx, y in enumerate(abs_y):
    plt.figure(1)
    plt.title("归一化单边振幅谱 channel" + str(idx+1))
    plt.xlabel("Frequency")  # Text for X-Axis
    plt.plot(x[0:100], y[0: 100])
    plt.show()
