import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
import os
from definitions import DATA_ROOT
import mne
from scipy import stats
import statsmodels.api as sm
from statsmodels.distributions.mixture_rvs import mixture_rvs

# Plot a 1D density example
edf_path = os.path.join(DATA_ROOT, "imfs.edf")
np.random.seed(1)
raw = mne.io.read_raw_edf(edf_path, preload=True)
raw = raw.load_data()
data = raw._data
X = data[0][:, np.newaxis]
X_plot = np.linspace(X.min() * 1.5, X.max() * 1.5, X.shape[0])[:, np.newaxis]

true_dens = 0.3 * norm(0, 1).pdf(X_plot[:, 0]) + 0.7 * norm(5, 1).pdf(X_plot[:, 0])

fig, ax = plt.subplots()
ax.fill(X_plot[:, 0], true_dens, fc="black", alpha=0.2, label="input distribution")
colors = ["navy", "cornflowerblue", "darkorange"]
kernels = ["gaussian", "tophat", "epanechnikov"]
lw = 2

for color, kernel in zip(colors, kernels):
    kde = KernelDensity(kernel=kernel, bandwidth=1).fit(X)
    log_dens = kde.score_samples(X_plot)
    exp = np.exp(log_dens)
    ax.plot(
        X_plot[:, 0],
        np.exp(log_dens),
        color=color,
        lw=lw,
        linestyle="-",
        label="kernel = '{0}'".format(kernel),
    )

# ax.text(6, 0.38, "N={0} points".format(N))

ax.legend(loc="upper left")
ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), "+k")

# ax.set_xlim(-4, 9)
# ax.set_ylim(-0.02, 0.4)
plt.show()
a=1
