from pprint import pprint

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.integrate import simps
import yasa

data = np.loadtxt('data.txt')
sns.set(font_scale=1.2)


sf = 100.
time = np.arange(data.size) / sf
win = 4 * sf
freqs, psd = signal.welch(data[0:400], sf, nperseg=win)

# Plot the power spectrum
sns.set(font_scale=1.2, style='white')
plt.figure(figsize=(8, 6))
plt.plot(freqs, psd, color='k', lw=2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (V^2 / Hz)')
plt.ylim([0, psd.max() * 1.1])
plt.title("Welch's periodogram")
plt.xlim([0, freqs.max()])
sns.despine()

# Define delta lower and upper limits
low, high = 0.5, 4
plt.show()

res = yasa.bandpower_from_psd(psd, freqs)
r_res = res.iloc[:, 1:8]
r_res.columns = [
    'delta_power_ratio', 'theta_power_ratio',
    'alpha_power_ratio','sigma_power_ratio',
    'beta_power_ratio', 'gamma_power_ratio',
    'total_power'
]
pprint(r_res)