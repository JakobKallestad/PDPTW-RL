import pickle
import numpy as np

with open('pdp/pdp20_TEST1_seed1234.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data[0])