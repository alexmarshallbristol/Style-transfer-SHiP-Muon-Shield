
import numpy as np

import scipy.stats as stats

import glob

files = glob.glob('/eos/experiment/ship/user/amarshal/HUGE_GAN/h*.npy')

files = files[:100]

x_lim = 75
y_lim = 75

x = -1

for file in files:

	x += 1

	data = np.load(file)

	x_column = np.multiply(np.subtract(np.random.rand(np.shape(data)[0]),np.ones(np.shape(data)[0])*0.5)*2,np.ones(np.shape(data)[0])*x_lim)
	y_column = np.multiply(np.subtract(np.random.rand(np.shape(data)[0]),np.ones(np.shape(data)[0])*0.5)*2,np.ones(np.shape(data)[0])*y_lim)

	px_column = data[:,4]
	py_column = data[:,5]
	pz_column = data[:,6]

	weights = data[:,0]

	z_values = data[:,3]

	lower, upper = 0, 5
	mu, sigma = 1.5, 1
	X_pt = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
	X_pt = X_pt.rvs((2,np.shape(data)[0]))

	lower, upper = 0.25, 1.2
	mu, sigma = 1.1, 1
	X_pz = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
	X_pz = X_pz.rvs(np.shape(data)[0])

	px_column = px_column * X_pt[0]
	py_column = py_column * X_pt[1]
	pz_column = pz_column * X_pz


	generated_sample_full = [weights,x_column,y_column,z_values,px_column,py_column,pz_column]

	generated_sample_full = np.swapaxes(generated_sample_full,0,1)

	generated_sample_small = generated_sample_full[:100000]

	save = np.empty((0,7))

	for i in range(0, 99):
		save = np.append(save, generated_sample_small, axis=0)

	np.save('/eos/experiment/ship/user/amarshal/Ship_shield/blurred_huge_gen_data/geant_muons_100_times_blurred_%d'%x,save)
