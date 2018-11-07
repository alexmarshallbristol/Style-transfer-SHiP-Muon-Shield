
'''

G goes 6 -> 5

D goes 5 -> 1

'''

https://stackoverflow.com/questions/47947578/multi-input-models-using-keras-model-api



import numpy as np
from keras.datasets import mnist
from keras.layers import Input, Flatten, Dense, Reshape, Dropout
from keras.layers import BatchNormalization
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

_EPSILON = K.epsilon() # 10^-7 by default. Epsilon is used as a small constant to avoid ever dividing by zero. 

def _loss_generator(y_true, y_pred):
	y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
	out = -(K.log(y_pred))
	return K.mean(out, axis=-1)

class GAN(object):

	def __init__(self):

		parser = argparse.ArgumentParser()

		parser.add_argument('-l', action='store', dest='learning_rate', type=float,
		                    help='learning rate', default=0.0002)

		results = parser.parse_args()

		# self.width = width
		# self.height = height
		# self.channels = channels

		# self.shape = (self.width, self.height, self.channels) # Define shape of muon 'images'

		self.optimizerG = Adam(lr=results.learning_rate, beta_1=0.5, decay=0, amsgrad=True)
		self.optimizerD = Adam(lr=results.learning_rate, beta_1=0.5, decay=0, amsgrad=True)
		
		self.G = self.__generator()
		self.G.compile(loss=_loss_generator, optimizer=self.optimizerG)

		self.D = self.__discriminator()
		self.D.compile(loss='binary_crossentropy', optimizer=self.optimizerD, metrics=['accuracy'])

		self.stacked_generator_discriminator = self.__stacked_generator_discriminator()
		self.stacked_generator_discriminator.compile(loss=_loss_generator, optimizer=self.optimizerD)


	def __generator(self):

		model = Sequential()

		model.add(Dense(int(256), input_shape=(1,6)))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))

		model.add(Dense(int(512)))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
	
		model.add(Dense(5, activation='tanh'))
		model.add(Reshape((1,5)))

		# keras.layers.Concatenate(axis=-1)

		model.summary()

		return models

	def __discriminator(self):

		model = Sequential()

		model.add(Flatten(input_shape=(2,5)))
		model.add(Dense((int(256))))
		model.add(LeakyReLU(alpha=0.2))

		model.add(Dense((int(512))))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))

		model.add(Dense(1, activation='sigmoid'))

		model.summary()

		return model

	def __stacked_generator_discriminator(self):

		self.D.trainable = False

		model = Sequential()
		model.add(self.G)
		#is it here I should do the keras concatenate
		model.add(self.D)


		return model

	def train(self, epochs = 10000, batch = 50, save_interval = 100):

		d_loss_list = g_loss_list = np.empty(0)

		FairSHiP_sample = np.random.rand(1000,2,5)
		print(np.shape(FairSHiP_sample)[0],'samples from FairSHiP.')
		split = [0.7,0.3]
		print('Split:',split)
		split_index = int(split[0]*np.shape(FairSHiP_sample)[0])
		list_for_np_choice = np.arange(np.shape(FairSHiP_sample)[0]) 
		random_indicies = np.random.choice(list_for_np_choice, size=np.shape(FairSHiP_sample)[0], replace=False)
		training_sample = FairSHiP_sample[:split_index]
		test_sample = FairSHiP_sample[split_index:]
		list_for_np_choice = np.arange(np.shape(training_sample)[0]) 

		for e in range(epochs):

			random_indicies = np.random.choice(list_for_np_choice, size=(3,batch), replace=False)

			# Prepare training samples for D

			d_real_training = training_sample[random_indicies[0]]
			d_fake_training = training_sample[random_indicies[1]]

			random_dimension = np.expand_dims(np.expand_dims(np.random.rand(batch),1),1)
			d_fake_training_initial, throw_away = np.split(d_fake_training, [1], axis=1) # Remove the real final state information
			d_fake_training_initial_w_rand = np.concatenate((d_fake_training_initial, random_dimension),axis=2) # Add dimension of random noise

			print(np.shape(d_fake_training_initial_w_rand))
			synthetic_final_states = self.G.predict([d_fake_training_initial_w_rand]) # Run initial muon parameters through G for a final state guess

			d_fake_training_with_synthetic = np.concatenate((d_fake_training_initial, synthetic_final_states),axis=1) # Add inital state to final state guess
			
			legit_labels = np.ones((int(batch), 1)) # Create label arrays
			gen_labels = np.zeros((int(batch), 1))

			d_loss_legit = self.D.train_on_batch(d_real_training, legit_labels) # Train D
			d_loss_gen = self.D.train_on_batch(d_fake_training_with_synthetic, gen_labels)


			# Prepare training samples for G

			g_training = training_sample[random_indicies[2]]
			g_training, throw_away = np.split(g_training, [1], axis=1)
			random_dimension = np.expand_dims(np.expand_dims(np.random.rand(batch),1),1)
			g_training = np.concatenate((g_training, random_dimension),axis=2) # Add dimension of random noise

			y_mislabled = np.ones((batch, 1))

			print(np.shape(g_training))

			quit()
			g_loss = self.stacked_generator_discriminator.train_on_batch(g_training, y_mislabled)

			# Dosen't work - need to make a lambda layer that concatenates real inital data onto output of G in stacked GD.
			# Use keras concatenate




			print(g_loss)
			quit()






			print('Loading training file:',file,'...')

			X_train = np.load(file[0])

			muon_weights, X_train = np.split(X_train, [1], axis=1)

			random_indicies = np.random.choice(list_for_np_choice, size=int(batch/2), p=muon_weights, replace=False)
			legit_images = X_train[random_indicies].reshape(int(batch/2), self.width, self.height, self.channels)

			gen_noise = np.random.normal(0, 1, (int(batch/2), 100))
			syntetic_images = self.G.predict(gen_noise)

			legit_labels = np.ones((int(batch/2), 1))
			gen_labels = np.zeros((int(batch/2), 1))
			
			d_loss_legit = self.D.train_on_batch(legit_images, legit_labels)
			d_loss_gen = self.D.train_on_batch(syntetic_images, gen_labels)

			noise = np.random.normal(0, 1, (batch, 100))
			y_mislabled = np.ones((batch, 1))

			g_loss = self.stacked_generator_discriminator.train_on_batch(noise, y_mislabled)

			if cnt % int(save_interval/10) == 0: print ('Epoch: %d, [Discriminator :: d_loss: %f %f], [ Generator :: loss: %f]' % (cnt, d_loss_legit[0], d_loss_gen[0], g_loss))

			d_loss_list = np.append(d_loss_list, [[cnt,(d_loss_legit[0]+d_loss_gen[0])/2]], axis=0)
			g_loss_list = np.append(g_loss_list, [[cnt, g_loss]], axis=0)


			if cnt > 1 and cnt % save_interval == 0:

				self.plot_images()

	def plot_images(self):

		return 

if __name__ == '__main__':
	
	gan = GAN()

	gan.train()




