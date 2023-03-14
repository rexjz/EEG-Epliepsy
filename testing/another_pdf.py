import os

import matplotlib.pyplot as plt
import mne
import numpy as np
import statsmodels.api as sm

from definitions import DATA_ROOT

edf_path = os.path.join(DATA_ROOT, "imfs.edf")
np.random.seed(1)
raw = mne.io.read_raw_edf(edf_path, preload=True)
raw = raw.load_data()
obs_dist = raw._data[0]
x_lim = obs_dist.min() * 1.5, obs_dist.max() * 1.5

kde = sm.nonparametric.KDEUnivariate(obs_dist)
kde.fit(bw=0.000000001)  # Estimate the densities


fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111)

# # Plot the histogram
# ax.hist(
#     obs_dist,
#     bins=20,
#     density=True,
#     label="Histogram from samples",
#     zorder=5,
#     edgecolor="k",
#     alpha=0.5,
# )

# Plot the KDE as fitted using the default arguments
ax.plot(kde.support, kde.density, lw=3, label="KDE from samples", zorder=10)
density = kde.evaluate(obs_dist)
log_density = np.log(density)


# Plot the samples
ax.scatter(
    obs_dist,
    np.abs(np.random.randn(obs_dist.size)) / 40,
    marker="x",
    color="red",
    zorder=20,
    label="Samples",
    alpha=0.5,
)

ax.legend(loc="best")
ax.grid(True, zorder=-5)
ax.set_xlim(x_lim)
plt.show()
