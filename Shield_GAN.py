import numpy as np
from keras.datasets import mnist
from keras.layers import Input, Flatten, Dense, Reshape, Dropout, UpSampling2D, Convolution2D
from keras.layers import BatchNormalization, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
import random
import math
import matplotlib as mpl
from keras.models import load_model
from scipy.stats import chisquare
import os
mpl.use('TkAgg') 
mpl.use('Agg')
import matplotlib.pyplot as plt
import time
from keras import backend as K
import argparse
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from matplotlib.colors import LogNorm
import glob
from keras.models import Model

_EPSILON = K.epsilon() # 10^-7 by default. Epsilon is used as a small constant to avoid ever dividing by zero. 

def _loss_generator(y_true, y_pred):
	y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
	out = -(K.log(y_pred))
	return K.mean(out, axis=-1)

parser = argparse.ArgumentParser()

parser.add_argument('-l', action='store', dest='learning_rate', type=float,
                    help='learning rate', default=0.0002)

results = parser.parse_args()


# Define optimizers ...

optimizerG = Adam(lr=results.learning_rate, beta_1=0.5, decay=0, amsgrad=True)
optimizerD = Adam(lr=results.learning_rate, beta_1=0.5, decay=0, amsgrad=True)


# Build Generative model ...

initial_state_w_noise = Input(shape=(1,6))
# initial_state_w_noise = Input(shape=(1,5))

inital_state = Input(shape=(1,5))

H = Dense(512)(initial_state_w_noise)
H = LeakyReLU(alpha=0.2)(H)
H = BatchNormalization(momentum=0.8)(H)

H = Dense(512)(H)
H = LeakyReLU(alpha=0.2)(H)
H = BatchNormalization(momentum=0.8)(H)

H = Dense(512)(H)
H = LeakyReLU(alpha=0.2)(H)
H = BatchNormalization(momentum=0.8)(H)

H = Dense(40)(H)
H = LeakyReLU(alpha=0.2)(H)
H = BatchNormalization(momentum=0.8)(H)

H = Dense(5, activation='tanh')(H)
final_state_guess = Reshape((1,5))(H)

g_output = concatenate([inital_state, final_state_guess],axis=1)

generator = Model(inputs=[initial_state_w_noise,inital_state], outputs=[g_output])

generator.compile(loss=_loss_generator, optimizer=optimizerG)
generator.summary()



# Build Discriminator model ...

d_input = Input(shape=(2,5))

H = Flatten()(d_input)

H = Dense(512)(H)
H = LeakyReLU(alpha=0.2)(H)

H = Dense(512)(H)
H = LeakyReLU(alpha=0.2)(H)
H = Dropout(0.25)(H)

H = Dense(512)(H)
H = LeakyReLU(alpha=0.2)(H)
H = Dropout(0.25)(H)

H = Dense(40)(H)
H = LeakyReLU(alpha=0.2)(H)
H = Dropout(0.25)(H)

d_output = Dense(1, activation='sigmoid')(H)

discriminator = Model(d_input, d_output)

discriminator.compile(loss='binary_crossentropy',optimizer=optimizerD)
discriminator.summary()



# Build stacked GAN model ...

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

make_trainable(discriminator, False)

initial_state_w_noise = Input(shape=(1,6))
# initial_state_w_noise = Input(shape=(1,5))

inital_state = Input(shape=(1,5))

H = generator([initial_state_w_noise,inital_state])

gan_output = discriminator(H)

GAN_stacked = Model(inputs=[initial_state_w_noise,inital_state], outputs=[gan_output])
GAN_stacked.compile(loss=_loss_generator, optimizer=optimizerD)
GAN_stacked.summary()



# Define arrays to save loss data ...

d_loss_list = g_loss_list = np.empty(0)


# Define training variables ...

epochs = 100000000000000000000

batch = 50

test_batch = 25000
save_interval = 1000

load_new_training_data = 10000
number_of_files_in_folder = 25

# def pre_process(input_array):

# 	'''
	
# 	Dummy pre process array. In reality this step will be completed before np.load()

# 	'''

# 	for i, row in enumerate(input_array):
# 		for j in range(0, 5):
# 			row[0][j] = (row[0][j] - 0.5) * 2

# 		for j in range(0, 5):
# 			row[1][j] = (row[0][j]/2)**2 + 0.25

# 	return input_array


# # Load data ...

# FairSHiP_sample = np.random.rand(100000,2,5) # In reality will use np.load() here 

# FairSHiP_sample = pre_process(FairSHiP_sample)

# FairSHiP_sample = np.load('/mnt/storage/scratch/am13743/SHIP_SHIELD/simulated_array_pre_processed.npy')
# FairSHiP_sample = np.load('/Users/am13743/Desktop/style-transfer-GANs/data/merged_files.npy')

# FairSHiP_sample = np.load('merged_files.npy')


# print(np.shape(FairSHiP_sample)[0],'samples from FairSHiP.')

# FairSHiP_sample  = FairSHiP_sample[:10000000]

# split = [0.7,0.3]
# print('Split:',split)
# split_index = int(split[0]*np.shape(FairSHiP_sample)[0])
# list_for_np_choice = np.arange(np.shape(FairSHiP_sample)[0]) 
# random_indicies = np.random.choice(list_for_np_choice, size=np.shape(FairSHiP_sample)[0], replace=False)
# training_sample = FairSHiP_sample[:split_index]
# test_sample = FairSHiP_sample[split_index:]

# print('Training:',np.shape(training_sample), 'Test:',np.shape(test_sample))

# list_for_np_choice = np.arange(np.shape(training_sample)[0]) 
# list_for_np_choice_test = np.arange(np.shape(test_sample)[0]) 



# quit()

# Start training loop ...

for e in range(epochs):

	if e % load_new_training_data == 0:

		file_index = np.random.randint(0, number_of_files_in_folder)
	
		FairSHiP_sample = np.load('/eos/experiment/ship/user/amarshal/Ship_shield/training_files/geant_%d.npy'%file_index)

		print('Loading new file, at training step',e,'- new file has',np.shape(FairSHiP_sample)[0],'samples from FairSHiP.')

		split = [0.8,0.2]

		print('Split:',split)
		split_index = int(split[0]*np.shape(FairSHiP_sample)[0])
		list_for_np_choice = np.arange(np.shape(FairSHiP_sample)[0]) 
		random_indicies = np.random.choice(list_for_np_choice, size=np.shape(FairSHiP_sample)[0], replace=False)
		training_sample = FairSHiP_sample[:split_index]
		test_sample = FairSHiP_sample[split_index:]

		print('Training:',np.shape(training_sample), 'Test:',np.shape(test_sample))

		list_for_np_choice = np.arange(np.shape(training_sample)[0]) 
		list_for_np_choice_test = np.arange(np.shape(test_sample)[0]) 


	random_indicies = np.random.choice(list_for_np_choice, size=(3,batch), replace=False)

	# Prepare training samples for D ...

	d_real_training = training_sample[random_indicies[0]]
	d_fake_training = training_sample[random_indicies[1]]

	# random_dimension = np.expand_dims(np.expand_dims(np.random.rand(batch),1),1)
	random_dimension = np.expand_dims(np.expand_dims(np.random.normal(0,0.01,batch),1),1)
	d_fake_training_initial, throw_away = np.split(d_fake_training, [1], axis=1) # Remove the real final state information
	d_fake_training_initial_w_rand = np.concatenate((d_fake_training_initial, random_dimension),axis=2) # Add dimension of random noise

	synthetic_output = generator.predict([d_fake_training_initial_w_rand, d_fake_training_initial]) # Run initial muon parameters through G for a final state guess and initial state in shape (2,5)

	legit_labels = np.ones((int(batch), 1)) # Create label arrays
	gen_labels = np.zeros((int(batch), 1))

	d_loss_legit = discriminator.train_on_batch(d_real_training, legit_labels) # Train D
	d_loss_gen = discriminator.train_on_batch(synthetic_output, gen_labels)

	# Prepare training samples for G ...

	g_training = training_sample[random_indicies[2]]
	g_training, throw_away = np.split(g_training, [1], axis=1)
	# random_dimension = np.expand_dims(np.expand_dims(np.random.rand(batch),1),1)
	random_dimension = np.expand_dims(np.expand_dims(np.random.normal(0,0.01,batch),1),1)
	g_training_w_noise = np.concatenate((g_training, random_dimension),axis=2) # Add dimension of random noise

	y_mislabled = np.ones((batch, 1))

	g_loss = GAN_stacked.train_on_batch([g_training_w_noise, g_training], y_mislabled)

	if e % 100 == 0 and e > 1: 
		print('Step:',e)


	if e % save_interval == 0 and e > 1: 

		print('Saving',e,'...')

		random_indicies = np.random.choice(list_for_np_choice_test, size=test_batch, replace=False)

		sample_to_test = test_sample[random_indicies]

		# random_dimension = np.expand_dims(np.expand_dims(np.random.rand(test_batch),1),1)
		random_dimension = np.expand_dims(np.expand_dims(np.random.normal(0,0.01,test_batch),1),1)
		sample_to_test_initial, throw_away = np.split(sample_to_test, [1], axis=1) # Remove the real final state information
		sample_to_test_initial_w_rand = np.concatenate((sample_to_test_initial, random_dimension),axis=2) # Add dimension of random noise

		synthetic_test_output = generator.predict([sample_to_test_initial_w_rand, sample_to_test_initial]) # Run initial muon parameters through G for a final state guess and initial state in shape (2,5)

		print(np.shape(sample_to_test), np.shape(synthetic_test_output))

		# create BDT between generated and real output
		# train on some, test on others - always randomise what to test/train on


		# def post_process(input_array):

		# WRONG _ DOESNT DO -1 to 1

		# 	input_array = np.reshape(input_array,(2,np.shape(input_array)[0],5))

		# 	x_i = input_array[0,:,0]
		# 	y_i = input_array[0,:,1]
		# 	px_i = input_array[0,:,2]
		# 	py_i = input_array[0,:,3]
		# 	pz_i = input_array[0,:,4]

		# 	def un_process(input_array_2, minimum, full_range):
		# 		# input_array_2 += minimum * (-1)
		# 		# input_array = input_array/full_range

		# 		input_array_2 = input_array_2*full_range
		# 		input_array_2 += minimum

		# 		return input_array_2

		# 	x_i = un_process(x_i, -2500, 5000)
		# 	y_i = un_process(y_i, -2500, 5000)
		# 	px_i = un_process(px_i, -20, 40)
		# 	py_i = un_process(py_i, -20, 40)
		# 	pz_i = un_process(pz_i, 0, 400)

		# 	inital = [x_i, y_i, px_i, py_i, pz_i]

		# 	x_f = input_array[1,:,0]
		# 	y_f = input_array[1,:,1]
		# 	px_f = input_array[1,:,2]
		# 	py_f = input_array[1,:,3]
		# 	pz_f = input_array[1,:,4]

		# 	x_f = un_process(x_f, -2500, 5000)
		# 	y_f = un_process(y_f, -2500, 5000)
		# 	px_f = un_process(px_f, -20, 40)
		# 	py_f = un_process(py_f, -20, 40)
		# 	pz_f = un_process(pz_f, 0, 400)

		# 	final = [x_f, y_f, px_f, py_f, pz_f]

		# 	input_array = [inital, final]

		# 	input_array = np.reshape(input_array,(np.shape(input_array)[2],2,5))

		# 	return input_array

		# sample_to_test = post_process(sample_to_test)
		# synthetic_test_output = post_process(synthetic_test_output)

		def plot_2d_hists(dim_1, dim_2):

			plt.figure(figsize=(8,8))

			plt.subplot(3,2,1)
			#inital real
			# plt.hist2d(sample_to_test[:,0,dim_1], sample_to_test[:,0,dim_2], bins=50, range=[[-1,1],[-1,1]],norm=LogNorm())
			plt.hist2d(sample_to_test[:,0,dim_1], sample_to_test[:,0,dim_2], bins=50,range=[[np.amin(sample_to_test[:,0,dim_1]),np.amax(sample_to_test[:,0,dim_1])],[np.amin(sample_to_test[:,0,dim_2]),np.amax(sample_to_test[:,0,dim_2])]],norm=LogNorm())
			plt.subplot(3,2,2)
			#final real
			# plt.hist2d(sample_to_test[:,1,dim_1], sample_to_test[:,1,dim_2], bins=50, range=[[-1,1],[-1,1]],norm=LogNorm())
			plt.hist2d(sample_to_test[:,1,dim_1], sample_to_test[:,1,dim_2], bins=50,range=[[np.amin(sample_to_test[:,1,dim_1]),np.amax(sample_to_test[:,1,dim_1])],[np.amin(sample_to_test[:,1,dim_2]),np.amax(sample_to_test[:,1,dim_2])]],norm=LogNorm())

			plt.subplot(3,2,3)
			#inital fake
			# plt.hist2d(synthetic_test_output[:,0,dim_1], synthetic_test_output[:,0,dim_2], bins=50, range=[[-1,1],[-1,1]],norm=LogNorm())
			plt.hist2d(synthetic_test_output[:,0,dim_1], synthetic_test_output[:,0,dim_2], bins=50,range=[[np.amin(sample_to_test[:,0,dim_1]),np.amax(sample_to_test[:,0,dim_1])],[np.amin(sample_to_test[:,0,dim_2]),np.amax(sample_to_test[:,0,dim_2])]],norm=LogNorm())
			plt.subplot(3,2,4)
			#final fake
			# plt.hist2d(synthetic_test_output[:,1,dim_1], synthetic_test_output[:,1,dim_2], bins=50, range=[[-1,1],[-1,1]],norm=LogNorm())
			plt.hist2d(synthetic_test_output[:,1,dim_1], synthetic_test_output[:,1,dim_2], bins=50,range=[[np.amin(sample_to_test[:,1,dim_1]),np.amax(sample_to_test[:,1,dim_1])],[np.amin(sample_to_test[:,1,dim_2]),np.amax(sample_to_test[:,1,dim_2])]],norm=LogNorm())
			plt.subplot(3,2,5)
			#final fake
			plt.hist2d(synthetic_test_output[:,1,dim_1], synthetic_test_output[:,1,dim_2], bins=50, range=[[-1,1],[-1,1]],norm=LogNorm())
			# plt.hist2d(synthetic_test_output[:,1,dim_1], synthetic_test_output[:,1,dim_2], bins=50,range=[[np.amin(synthetic_test_output[:,0,dim_1]),np.amax(synthetic_test_output[:,0,dim_1])],[np.amin(synthetic_test_output[:,0,dim_2]),np.amax(synthetic_test_output[:,0,dim_2])]],norm=LogNorm())
			
			# plt.savefig('/mnt/storage/scratch/am13743/SHIP_SHIELD/checkpoint_%d_%d.png'%(dim_1,dim_2))
			# plt.savefig('/Users/am13743/Desktop/style-transfer-GANs/data/plots/checkpoint_%d_%d.png'%(dim_1,dim_2))
			plt.savefig('checkpoint_%d_%d.png'%(dim_1,dim_2))

			plt.close('all')

		for first in range(0, 5):
			for second in range(first+1, 5):
				plot_2d_hists(first, second)





