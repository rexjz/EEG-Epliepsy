import numpy as np

TIME_DOMAIN_FEATURE = ['mean', 'var', 'std', 'rms', 'skew', 'kurt', 'abs_max', 'fuzzy_en', 'piece_index']
FREQUENCY_DOMAIN_FEATURE = ['theta_power_ratio',
                            'delta_power_ratio',
                            'alpha_power_ratio',
                            'beta_power_ratio',
                            'gamma_power_ratio',
                            'piece_index']
SAMPLING_FREQUENCY = 500
# [inclusive, exclusive]
THETA_FREQ_RANGE = np.array([0, 4])
# [inclusive, exclusive]
DELTA_FREQ_RANGE = np.array([4, 8])
# [inclusive, exclusive]
ALPHA_FREQ_RANGE = np.array([8, 13])
# [inclusive, exclusive]
BETA_FREQ_RANGE = np.array([13, 30])


# return [inclusive, exclusive]
def GAMMA_FREQ_RANGE(n):
    n = int(n)
    return np.array([30, n])
