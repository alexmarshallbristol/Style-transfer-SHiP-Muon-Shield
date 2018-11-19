
import numpy as np

import scipy.stats as stats

import glob

import math

x_lim = 100
y_lim = 100

for x in range(0, 100):

	print(x)

	save_array = np.empty((0,5))

	size = 50000

	x_column = np.multiply(np.subtract(np.random.rand(size),np.ones(size)*0.5)*2,np.ones(size)*x_lim)
	y_column = np.multiply(np.subtract(np.random.rand(size),np.ones(size)*0.5)*2,np.ones(size)*y_lim)

	pz_column = np.multiply(np.random.rand(size),np.ones(size)*450)
	pt_column = np.multiply(np.random.rand(size),np.ones(size)*20)

	angle = np.multiply(np.random.rand(size),np.ones(size)*2*math.pi)

	px = np.cos(angle)*pt_column
	py = np.sin(angle)*pt_column

	bound = (-1)*(20.0/450.0)*pz_column + 20

	where = np.greater(pt_column,bound)

	generated_sample_full = [x_column,y_column,px,py,pz_column]
	generated_sample_full = np.delete(generated_sample_full,np.where(where == True),axis=1)
	generated_sample_full = np.swapaxes(generated_sample_full,0,1)

	for y in range(0, 100):
		save_array = np.append(save_array, generated_sample_full,axis=0)
	print(np.shape(save_array))

	np.save('/eos/experiment/ship/user/amarshal/Ship_shield/uniform_muons/uniform_blur_multiple_%d'%x,save_array)

print('Done ...')