import numpy as np
from slicing import *
from util.FuzzyEntropy import *
from EntropyHub._FuzzEn import FuzzEn
import math

record = sliced_data()[0]
[normal_record_pieces, seizure_record_pieces] = record
fuzzyEntropy = FuzzyEntropy(m=2, r_coff=0.15, n=2)
piece = np.concatenate([np.random.rand(100), np.random.rand(100)])
piece2 = np.concatenate([np.random.rand(100), 10 * np.array(np.random.rand(100))])
piece3 = np.concatenate([np.random.rand(20), np.array(range(30)), np.random.rand(50)*100, np.random.rand(50)*0.1, np.random.rand(50)*1000])
piece4 = normal_record_pieces[0][2]
piece5 = seizure_record_pieces[0][2]
# a = fuzzyEntropy.apply(piece)
a = [round((0.1 + piece.std()*0.25) / 2), round((0.1 + piece2.std()*0.25) / 2), round((0.1 + piece3.std()*0.25) / 2)]
b = [piece.std(),piece2.std(), piece3.std()]
(a_1,*_) = FuzzEn(piece, r=(0.2, 2.0), tau=1, Logx=math.e)
# b = fuzzyEntropy.apply(piece2)
(b_1,*_) = FuzzEn(piece2, r=(0.2, (0.1 + piece2.std()*0.25) / 2))
# # c = fuzzyEntropy.apply(piece3)
(c_1,*_) = FuzzEn(piece3, r=(0.2, (0.1 + piece3.std()*0.25) / 2))
(d_1, *_) = FuzzEn(piece4, r=(0.2, (0.1 + piece4.std()*0.25) / 2))
(e_1, *_) = FuzzEn(piece5, r=(0.2, (0.1 + piece5.std()*0.25) / 2))
print("normal: ", a_1[0] - a_1[1])
print("chaotic: ", b_1[0] - b_1[1])
print("chaotic: ", c_1[0] - c_1[1])
print("normal: ", d_1[0] - d_1[1])
print("seizure: ", e_1[0] - e_1[1])