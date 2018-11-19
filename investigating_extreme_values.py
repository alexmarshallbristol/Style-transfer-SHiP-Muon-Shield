import numpy as np

from keras.layers import Input, Flatten, Dense, Reshape, Dropout, BatchNormalization, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.models import load_model, Model
from keras import backend as K

import math

import matplotlib as mpl
mpl.use('TkAgg') 
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import scipy.stats as stats

_EPSILON = K.epsilon() # 10^-7 by default. Epsilon is used as a small constant to avoid ever dividing by zero. 

def _loss_generator(y_true, y_pred):
	y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
	out = -(K.log(y_pred))
	return K.mean(out, axis=-1)

boundaries_x_range = [-2500, 2500]
boundaries_y_range = [-2500, 2500]
boundaries_px_range = [-20,20]
boundaries_py_range = [-20,20]
boundaries_pz_range = [0, 400]

def test_single_muon():

	generator = load_model('/Users/am13743/Desktop/style-transfer-GANs/models/Generator.h5', custom_objects = {'_loss_generator':_loss_generator})

	data = np.load('/Users/am13743/Desktop/style-transfer-GANs/data/training_files/geant_11.npy')

	print(np.shape(data))

	list_for_np_choice = np.arange(np.shape(data)[0])

	random_indicies = np.random.choice(list_for_np_choice, size=100000, replace=False)

	array_to_test = data[random_indicies,1,:]

	random_dimension = np.random.rand(np.shape(array_to_test)[0],3)

	# print(np.amin(random_dimension), np.amax(random_dimension))
	# quit()

	# lower, upper = -1, 1
	# mu, sigma = 0, 1
	# random_dimension = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
	# random_dimension = random_dimension.rvs((np.shape(array_to_test)[0],3))


	array_to_test_w_rand = np.concatenate((array_to_test, random_dimension),axis=1)

	array_to_test = np.expand_dims(array_to_test,1)
	array_to_test_w_rand = np.expand_dims(array_to_test_w_rand,1)

	print(np.shape(array_to_test_w_rand), np.shape(array_to_test))
	synthetic_output = generator.predict([array_to_test_w_rand, array_to_test]) 

	def post_process(gan_array):

		gan_array = (gan_array/2) + 0.5

		def post_process_inner(input_array, minimum, full_range):

			input_array = input_array * full_range
			input_array += minimum

			return input_array

		for array in [gan_array]:
			for i in range(0, 2):
				array[:,i,0] = post_process_inner(array[:,i,0], boundaries_x_range[0], boundaries_x_range[1]-boundaries_x_range[0])
				array[:,i,1] = post_process_inner(array[:,i,1], boundaries_y_range[0], boundaries_y_range[1]-boundaries_y_range[0])
				array[:,i,2] = post_process_inner(array[:,i,2], boundaries_px_range[0], boundaries_px_range[1]-boundaries_px_range[0])
				array[:,i,3] = post_process_inner(array[:,i,3], boundaries_py_range[0], boundaries_py_range[1]-boundaries_py_range[0])
				array[:,i,4] = post_process_inner(array[:,i,4], boundaries_pz_range[0], boundaries_pz_range[1]-boundaries_pz_range[0])

		return gan_array

	synthetic_output = post_process(synthetic_output)

	where = np.where(synthetic_output[:,1,0] < -500)

	print(where, np.shape(where))

	# print(np.shape(array_to_test_w_rand))

	array_to_test_split, randomness = np.split(array_to_test_w_rand,[5],axis=2)

	# print(array_to_test_split, array_to_test)

	# print(np.shape(array_to_test_split), np.shape(randomness))

	random_dimension_new = np.random.rand(np.shape(array_to_test_split)[0],3)


	# print(np.shape(array_to_test_split), np.shape(random_dimension_new))
	array_to_test_split = np.squeeze(array_to_test_split)

	array_to_test_w_rand_new = np.concatenate((array_to_test_split, random_dimension_new),axis=1)

	array_to_test_split = np.expand_dims(array_to_test_split,1)
	array_to_test_w_rand_new = np.expand_dims(array_to_test_w_rand_new,1)


	synthetic_output_new = generator.predict([array_to_test_w_rand_new, array_to_test_split]) 

	synthetic_output_new = post_process(synthetic_output_new)
	# run the same with different randomness, plot plot the indexs that were bad


	print(np.shape(synthetic_output[where,1,0]))



	plt.figure(figsize=(8,4))
	plt.subplot(1,3,1)
	plt.hist2d(np.squeeze(synthetic_output[where,1,0]),np.squeeze(synthetic_output[where,1,1]),bins=50,norm=LogNorm())
	plt.subplot(1,3,2)
	plt.hist2d(np.squeeze(synthetic_output_new[where,1,0]),np.squeeze(synthetic_output_new[where,1,1]),bins=50,norm=LogNorm())
	plt.subplot(1,3,3)
	plt.hist2d(np.squeeze(synthetic_output_new[where,1,0]),np.squeeze(synthetic_output_new[where,1,1]),bins=50,norm=LogNorm(),range=[[-20,170],[-80,80]])
	plt.savefig('test.png')
	plt.close('all')

	synthetic_output_new = synthetic_output_new[where]

	where_2 = np.where(synthetic_output_new[:,1,0] < -500)

	print(where_2, np.shape(where_2))

	quit()







	print(np.shape(synthetic_output))
	plt.figure(figsize=(16,8))
	plt.subplot(2,4,1)
	plt.hist2d(synthetic_output[:,0,0], synthetic_output[:,0,1],bins=50,norm=LogNorm())
	plt.xlabel('x')
	plt.ylabel('y')
	plt.subplot(2,4,2)
	plt.hist2d(synthetic_output[:,1,0], synthetic_output[:,1,1],bins=50,norm=LogNorm())
	plt.xlabel('x')
	plt.ylabel('y')
	plt.subplot(2,4,3)
	plt.hist2d(synthetic_output[:,0,2], synthetic_output[:,0,4],bins=50,norm=LogNorm())
	plt.xlabel('P_x')
	plt.ylabel('P_z')
	plt.subplot(2,4,4)
	plt.hist2d(synthetic_output[:,1,2], synthetic_output[:,1,4],bins=50,norm=LogNorm())
	plt.xlabel('P_x')
	plt.ylabel('P_z')

	synthetic_output = synthetic_output[np.where(synthetic_output[:,1,0] < -500)]
	random_dimension = random_dimension[np.where(synthetic_output[:,1,0] < -500)]


	plt.subplot(2,4,5)
	plt.hist2d(synthetic_output[:,0,0], synthetic_output[:,0,1],bins=50,norm=LogNorm())
	plt.xlabel('x')
	plt.ylabel('y')
	plt.subplot(2,4,6)
	plt.hist2d(synthetic_output[:,1,0], synthetic_output[:,1,1],bins=50,norm=LogNorm())
	plt.xlabel('x')
	plt.ylabel('y')
	plt.subplot(2,4,7)
	plt.hist2d(synthetic_output[:,0,2], synthetic_output[:,0,4],bins=50,norm=LogNorm())
	plt.xlabel('P_x')
	plt.ylabel('P_z')
	plt.subplot(2,4,8)
	plt.hist2d(synthetic_output[:,1,2], synthetic_output[:,1,4],bins=50,norm=LogNorm())
	plt.xlabel('P_x')
	plt.ylabel('P_z')
	plt.savefig('test.png')
	plt.close('all')

	plt.hist2d(random_dimension[:,1],random_dimension[:,2],bins=50,norm=LogNorm())
	plt.savefig('test_rand.png')
	plt.close('all')





	print(np.shape(synthetic_output))
	
	# plt.hist2d(synthetic_output[:,0,0], synthetic_output[:,0,1],bins=50,norm=LogNorm())
	# plt.savefig('here.png')

	# array_to_test = synthetic_output[:,0,:]
	# array_to_test_w_rand = np.concatenate((array_to_test, random_dimension),axis=1)
	# array_to_test = np.expand_dims(array_to_test,1)
	# array_to_test_w_rand = np.expand_dims(array_to_test_w_rand,1)

	location = np.where(synthetic_output[:,1,0] < -500)

	array_to_test_w_rand = array_to_test_w_rand[location]
	array_to_test = array_to_test[location]

	print(array_to_test_w_rand[0],array_to_test[0],'here1')
	synthetic_output = generator.predict([array_to_test_w_rand, array_to_test]) 
	synthetic_output = post_process(synthetic_output)

	plt.hist2d(synthetic_output[:,1,0], synthetic_output[:,1,1],bins=50,norm=LogNorm())
	plt.savefig('here.png')
	plt.close('all')

	plt.hist2d(array_to_test_w_rand[:,0,6], array_to_test_w_rand[:,0,7],bins=50,norm=LogNorm())
	plt.savefig('here_r.png')
	plt.close('all')


	array_to_test_w_rand = array_to_test_w_rand[location]
	array_to_test = array_to_test[location]

	print(array_to_test_w_rand[0],array_to_test[0],'here2')
	synthetic_output = generator.predict([array_to_test_w_rand, array_to_test]) 
	synthetic_output = post_process(synthetic_output)

	plt.hist2d(synthetic_output[:,1,0], synthetic_output[:,1,1],bins=50,norm=LogNorm())
	plt.savefig('here2.png')
	plt.close('all')

	return 

test_single_muon()

















