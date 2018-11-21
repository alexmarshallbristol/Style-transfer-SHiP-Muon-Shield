import numpy as np

import math

import matplotlib as mpl
mpl.use('TkAgg') 
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

dis = np.load('/Users/am13743/Desktop/style-transfer-GANs/data/training_files/geant_1.npy')

print(np.amin(dis[:,1,0]), np.amax(dis[:,1,0]))
print(np.amin(dis[:,1,1]), np.amax(dis[:,1,1]))
print(np.amin(dis[:,1,2]), np.amax(dis[:,1,2]))
print(np.amin(dis[:,1,3]), np.amax(dis[:,1,3]))
print(np.amin(dis[:,1,4]), np.amax(dis[:,1,4]))