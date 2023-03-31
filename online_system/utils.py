import numpy as np


def get_confidence_interval(x, confidence):
    values = [np.random.choice(x, size=len(x), replace=True).mean() for i in range(5000)]
    return np.percentile(values, [100 * (1 - confidence) / 2, 100 * (1 - (1 - confidence) / 2)])
