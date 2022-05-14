from scipy.io import loadmat
import os

label_root = 'SumMe/GT'
for filename in os.listdir(label_root):
    data = loadmat(filename)