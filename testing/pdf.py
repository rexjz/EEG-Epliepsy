import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import os
from definitions import DATA_ROOT

edf_path = os.path.join(DATA_ROOT, "imfs.edf")
np.random.seed(1)
raw = mne.io.read_raw_edf(edf_path, preload=True)
raw = raw.load_data()
data = raw._data
for i in range(0, data.shape[0]):
    tag = "imf " + str(i)
    X = data[i][:, np.newaxis]
    X_plot = np.linspace(-0.0001, 0.0001, 8000)[:, np.newaxis]
    kde = KernelDensity(kernel='gaussian', bandwidth=0.000005).fit(X)
    params = kde.get_params()
    log_Y_plot = kde.score_samples(X_plot)
    Y_plot = np.exp(log_Y_plot)
    plt.figure(i)
    plt.title(tag)
    plt.plot(X_plot, Y_plot, linestyle="-", color='black', label="ATT-RLSTM", linewidth=0.5, markersize=0.5)
    plt.plot(X[:, 0], -0.5 - 1 * np.random.random(X.shape[0]), "+k")
    plt.xlim(-0.00003, 0.00003)
    plt.show()

    log_density = kde.score_samples(X)
    density = np.exp(log_density)
    a = log_density * density
    entropy = -np.sum(log_density * density)
    print(tag + ": ", entropy)
