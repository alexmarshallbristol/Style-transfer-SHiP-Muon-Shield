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

	# get kinematics of muon from single muon plots on the sharelatex at 1085
	# pre-process this
	# run through 10000 times and make plots
	# compare to geant4 plots on the share latex

	data = np.load('/Users/am13743/Desktop/style-transfer-GANs/data/training_files/geant_11.npy')

	index = np.random.randint(0,np.shape(data)[0])

	# print(index)
	index = 685110

	array_to_test = np.empty((0,5))

	size_to_test = 10000

	for x in range(0, size_to_test):
		array_to_test = np.append(array_to_test,[data[index,0,:]],axis=0)

	# (50, 1, 5) (50, 1, 6)

	random_dimension = np.random.rand(size_to_test,3)

	print(np.shape(array_to_test), np.shape(random_dimension))

	array_to_test_w_rand = np.concatenate((array_to_test, random_dimension),axis=1)

	array_to_test = np.expand_dims(array_to_test,1)
	array_to_test_w_rand = np.expand_dims(array_to_test_w_rand,1)

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

	print(np.shape(synthetic_output))

	plt.figure(figsize=(12,4))
	plt.subplot(1,3,1)
	plt.hist2d(synthetic_output[:,1,0], synthetic_output[:,1,1],bins=50,norm=LogNorm(),range=[[40,50],[-5,5]])
	plt.xlabel('x')
	plt.ylabel('y')
	plt.subplot(1,3,2)
	plt.hist2d(synthetic_output[:,1,0], synthetic_output[:,1,1],bins=50,norm=LogNorm())
	plt.xlabel('x')
	plt.ylabel('y')
	plt.subplot(1,3,3)
	plt.hist2d(synthetic_output[:,1,2], synthetic_output[:,1,4],bins=50,norm=LogNorm())
	plt.xlabel('P_x')
	plt.ylabel('P_z')
	plt.savefig('test.png')

	return 

test_single_muon()