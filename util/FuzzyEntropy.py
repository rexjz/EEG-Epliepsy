import math
from tqdm import tqdm
import numpy as np
from pandas import *
from typing import List


def distance(v1, v2):
    return np.max(np.abs(v1 - v2))


class FuzzyEntropy:

    def __init__(self, *, m, n, r_coff):
        self.m = m
        self.n = n
        self.r_coff = r_coff
        self.r = None

    def fuzzy_function(self, *, d):
        return math.pow(math.e, -pow(d, self.n) / self.r)

    def apply(self, data: np.ndarray):
        self.r = np.std(data) * self.r_coff
        phi_1 = self.calculate_phi(data, self.m)
        phi_2 = self.calculate_phi(data, self.m + 1)
        return math.log(phi_1) - math.log(phi_2)

    def calculate_phi(self, data, m):
        print('calculate_phi m=', m)
        # slice
        vectors = []
        for i in range(0, len(data) - m + 1):
            raw_vector = data[i:i + m]
            vector = raw_vector - np.mean(raw_vector)
            vectors.append(vector)
        fuzzy_df = self.calculate_fuzzy_df(vectors)
        sum = fuzzy_df.values.sum()
        print('sum', sum)
        return sum / (fuzzy_df.shape[0] * fuzzy_df.shape[0] - fuzzy_df.shape[0])

    def calculate_fuzzy_df(self, vectors: List[np.array]):
        last_log = 0
        length = len(vectors)
        print(length)
        distance_df = DataFrame(index=range(0, len(vectors)), columns=range(0, len(vectors)))
        fuzzy_df = DataFrame(index=range(0, len(vectors)), columns=range(0, len(vectors)))
        progress_bar = tqdm(range(0, length-1), unit="row",desc="calculating fuzzy_df")
        for i in range(0, length):
            for j in range(0, length):
                if i == j:
                    continue
                if math.isnan(distance_df[i][j]):
                    distance_df[i][j] = distance(vectors[i], vectors[j])
                else:
                    distance_df[i][j] = distance_df[j][i]

                if math.isnan(fuzzy_df[i][j]):
                    fuzzy_df[i][j] = self.fuzzy_function(d=distance_df[i][j])
                else:
                    fuzzy_df[i][j] = fuzzy_df[j][i]
            progress_bar.update(1)
        progress_bar.close()
        for k in range(0, length):
            fuzzy_df[k][k] = 0
        return fuzzy_df

