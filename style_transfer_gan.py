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

	def __init__(self, width=1, height=6, channels=1):

		parser = argparse.ArgumentParser()

		parser.add_argument('-l', action='store', dest='learning_rate', type=float,
		                    help='learning rate', default=0.0002)

		results = parser.parse_args()

		self.width = width
		self.height = height
		self.channels = channels

		self.shape = (self.width, self.height, self.channels) # Define shape of muon 'images'

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

		model.add(Dense(int(256), input_shape=(6,)))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))

		model.add(Dense(int(512)))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
	
		model.add(Dense(self.width  * self.height * self.channels, activation='tanh'))
		model.add(Reshape((self.width, self.height, self.channels)))

		model.summary()

		return model

	def __discriminator(self):

		#discriminator takes in both inital and final in a concantenated array

		#dunno whether to concantenate side to side or one on top of another (probs one on top of another)

		#outputs probability of the result being accurate

		#training is completed by lots of real muon [input array, output array] concantenated arrays


		model = Sequential()

		model.add(Flatten(input_shape=self.shape * 2 (lol) ))
		model.add(Dense((int(256)), input_shape=self.shape))
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
		model.add(self.D)

		return model

	def train(self, dimensions, epochs=1E10, batch = 100, save_interval = 25000):

		# Initialize arrays to store progress information, losses and rchi2 values. 
		d_loss_list = g_loss_list = bdt_sum_overlap_list = bdt_rchi2_list_mom = bdt_rchi2_list = np.empty((0,2))

		t0 = time.time()

		# Start best_bdt_rchi2 value extremely high to be sure early values are lower.
		best_bdt_rchi2 = best_bdt_rchi2_mom = 1E30
		best_bdt_overlap = 0


		bdt_rchi2_list_every_10 = bdt_rchi2_list_every_10_mom = np.empty(0)
		bdt_rchi2_list_every_10 = np.append(bdt_rchi2_list_every_10, 1E30)
		bdt_rchi2_list_every_10_mom = np.append(bdt_rchi2_list_every_10_mom, 1E30)
		bdt_rchi2_list_every_10_save_name = bdt_rchi2_list_every_10_save_name_mom = 1

		list_of_training_files = glob.glob('%s*%s'%(location_of_training_files,training_files))

		for cnt in range(epochs):

			if cnt > 249000: 
				save_interval = 5000

			if cnt % save_interval == 0: 

				file = np.random.choice(list_of_training_files, 1)

				print('Loading training file:',file,'...')

				X_train = np.load(file[0])

				muon_weights, X_train = np.split(X_train, [1], axis=1)

				muon_weights = np.squeeze(muon_weights/np.sum(muon_weights))

				# Simple list of ordered integers that will be fed to np.random.choice (only takes 1D array)
				# This list of ints is alligned with muon_weights array.
				# This allows weight corrected sampling.
				list_for_np_choice = np.arange(np.shape(X_train)[0]) 



			# Previous un-weighted sampling.
			# random_index = np.random.randint(0, len(X_train) - batch/2)
			# legit_images = X_train[random_index : random_index + int(batch/2)].reshape(int(batch/2), self.width, self.height, self.channels)

			# Weighted sampling.
			random_indicies = np.random.choice(list_for_np_choice, size=int(batch/2), p=muon_weights, replace=False)
			legit_images = X_train[random_indicies].reshape(int(batch/2), self.width, self.height, self.channels)

			# Generate sample of fake images. Initially when G knows nothing this will just be random noise.
			gen_noise = np.random.normal(0, 1, (int(batch/2), 100))
			syntetic_images = self.G.predict(gen_noise)

			# Label real images with 1, and fake images with 0.
			legit_labels = np.ones((int(batch/2), 1))
			gen_labels = np.zeros((int(batch/2), 1))

			# Add normal noise to labels, need to understand if this helps reduce chance of mode collapse.
			for i in range(0, len(legit_labels)):
				legit_labels[i] = legit_labels[i] + np.random.normal(0,0.3)
			for i in range(0, len(gen_labels)):
				gen_labels[i] = gen_labels[i] + np.random.normal(0,0.3)
			
			d_loss_legit = self.D.train_on_batch(legit_images, legit_labels)
			d_loss_gen = self.D.train_on_batch(syntetic_images, gen_labels)

			# Feed latent noise to stacked_generator_discriminator for training. Misslabel it (unsure if this affects anything). 
			noise = np.random.normal(0, 1, (batch, 100))
			y_mislabled = np.ones((batch, 1))

			g_loss = self.stacked_generator_discriminator.train_on_batch(noise, y_mislabled)

			if cnt % 1000 == 0: print ('Epoch: %d, [Discriminator :: d_loss: %f %f], [ Generator :: loss: %f]' % (cnt, d_loss_legit[0], d_loss_gen[0], g_loss))

			d_loss_list = np.append(d_loss_list, [[cnt,(d_loss_legit[0]+d_loss_gen[0])/2]], axis=0)
			g_loss_list = np.append(g_loss_list, [[cnt, g_loss]], axis=0)


			if cnt > 1 and cnt % save_interval == 0:
	

				if len(bdt_rchi2_list_every_10) > 10 and cnt > 250000:#rest every 10
					bdt_rchi2_list_every_10 = np.empty(0)
					bdt_rchi2_list_every_10 = np.append(bdt_rchi2_list_every_10, 1E30)
					bdt_rchi2_list_every_10_save_name += 1


				if len(bdt_rchi2_list_every_10_mom) > 10 and cnt > 250000:#rest every 10
					bdt_rchi2_list_every_10_mom = np.empty(0)
					bdt_rchi2_list_every_10_mom = np.append(bdt_rchi2_list_every_10_mom, 1E30)
					bdt_rchi2_list_every_10_save_name_mom += 1


				plt.subplot(1,2,1)
				plt.title('Discriminator loss')
				plt.plot(d_loss_list[:,0],d_loss_list[:,1])
				plt.subplot(1,2,2)
				plt.title('Generator loss')
				plt.plot(g_loss_list[:,0],g_loss_list[:,1])
				plt.savefig('%s%s/test_output/loss.png'%(storage_location,file_number))
				plt.close('all')

				bdt_rchi2_list, bdt_sum_overlap_list, bdt_rchi2_list_mom = self.plot_images(dimensions,bdt_rchi2_list_mom, X_train ,bdt_rchi2_list, t0, bdt_sum_overlap_list, list_for_np_choice, muon_weights, save2file=True, step=cnt)

				if bdt_rchi2_list[-1][1] < best_bdt_rchi2:
					print('Saving best rchi2.')
					with open("%s%s/test_output/models/best_bdt_rchi2.txt"%(storage_location,file_number), "a") as myfile:
						myfile.write('\n %d, %.3f'%(cnt, bdt_rchi2_list[-1][1]))
					self.G.save('%s%s/test_output/models/Generator_%s_best_rchi2.h5'%(storage_location,file_number,save_name))
					best_bdt_rchi2 = bdt_rchi2_list[-1][1]

				if bdt_rchi2_list_mom[-1][1] < best_bdt_rchi2_mom:
					print('Saving best rchi2.')
					with open("%s%s/test_output/models/best_bdt_rchi2_mom.txt"%(storage_location,file_number), "a") as myfile:
						myfile.write('\n %d, %.3f'%(cnt, bdt_rchi2_list_mom[-1][1]))
					self.G.save('%s%s/test_output/models/Generator_%s_best_rchi2_mom.h5'%(storage_location,file_number,save_name))
					best_bdt_rchi2_mom = bdt_rchi2_list_mom[-1][1]

				if bdt_sum_overlap_list[-1][1] > best_bdt_overlap:
					print('Saving best overlap.')
					with open("%s%s/test_output/models/best_bdt_overlap.txt"%(storage_location,file_number), "a") as myfile:
						myfile.write('\n %d, %.3f'%(cnt, bdt_sum_overlap_list[-1][1]))
					self.G.save('%s%s/test_output/models/Generator_%s_best_overlap.h5'%(storage_location,file_number,save_name))
					best_bdt_overlap = bdt_sum_overlap_list[-1][1]

				self.G.save('%s%s/test_output/models/Generator_%s_current.h5'%(storage_location,file_number,save_name))
				self.D.save('%s%s/test_output/models/Discriminator_%s_current.h5'%(storage_location,file_number,save_name))

				if bdt_rchi2_list[-1][1] < np.amin(bdt_rchi2_list_every_10):
					print('Saving best rchi2.')
					with open("%s%s/test_output/models/best_bdt_rchi2_best_every_10.txt"%(storage_location,file_number), "a") as myfile:
						myfile.write('\n %d %d, %.3f'%(bdt_rchi2_list_every_10_save_name,cnt, bdt_rchi2_list[-1][1]))
					self.G.save('%s%s/test_output/models/Generator_%s_best_rchi2_best_every_10_%d.h5'%(storage_location,file_number,save_name,bdt_rchi2_list_every_10_save_name))
				bdt_rchi2_list_every_10 = np.append(bdt_rchi2_list_every_10, bdt_rchi2_list[-1][1])

				if bdt_rchi2_list_mom[-1][1] < np.amin(bdt_rchi2_list_every_10_mom):
					print('Saving best rchi2_mom.')
					with open("%s%s/test_output/models/best_bdt_rchi2_best_every_10_mom.txt"%(storage_location,file_number), "a") as myfile:
						myfile.write('\n %d %d, %.3f'%(bdt_rchi2_list_every_10_save_name_mom,cnt, bdt_rchi2_list_mom[-1][1]))
					self.G.save('%s%s/test_output/models/Generator_%s_best_rchi2_best_every_10_%d_mom.h5'%(storage_location,file_number,save_name,bdt_rchi2_list_every_10_save_name_mom))
				bdt_rchi2_list_every_10_mom = np.append(bdt_rchi2_list_every_10_mom, bdt_rchi2_list_mom[-1][1])

	def plot_images(self, dimensions, bdt_rchi2_list_mom, X_train, bdt_rchi2_list, t0, bdt_sum_overlap_list, list_for_np_choice, muon_weights, save2file=False, samples=16, step=0):

		noise_size = 250000

		bdt_train_size = 50000

		noise = np.random.normal(0, 1, (noise_size, 100))

		images = self.G.predict(noise)

		# Initialise BDT classifier used to check progress.
		clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)

		# Create training samples for BDT.
		random_indicies = np.random.choice(list_for_np_choice, size=(noise_size), p=muon_weights, replace=False)

		real_training_data = X_train[random_indicies[:50000]]

		real_test_data = X_train[random_indicies[50000:]]

		fake_training_data = np.squeeze(images[:bdt_train_size])

		fake_test_data = np.squeeze(images[bdt_train_size:])

		real_training_labels = np.ones(bdt_train_size)

		fake_training_labels = np.zeros(bdt_train_size)

		total_training_data = np.concatenate((real_training_data, fake_training_data))

		total_training_labels = np.concatenate((real_training_labels, fake_training_labels))

		clf.fit(total_training_data, total_training_labels)

		out_real = clf.predict_proba(real_test_data)

		out_fake = clf.predict_proba(fake_test_data)


		plt.hist([out_real[:,1],out_fake[:,1]], bins = 100,label=['real','gen'], histtype='step')
		plt.xlabel('Output of BDT')
		plt.legend(loc='upper right')
		plt.savefig('%s%s/test_output/bdt/BDT_out_%d.png'%(storage_location,file_number,step), bbox_inches='tight')
		plt.close('all')

		# Create histograms from the output of BDT.
		real_hist = np.histogram(out_real[:,1],bins=100,range=[0,1])
		fake_hist = np.histogram(out_fake[:,1],bins=100,range=[0,1])

		# Compute rChi2 and overlap values for these histogram distrbiutions.

		bdt_chi2 = 0
		dof = 0

		for bin_index in range(0, 100):
			if real_hist[0][bin_index] > 9 and fake_hist[0][bin_index] > 9:
				dof += 1
				bdt_chi2 += ((real_hist[0][bin_index] - fake_hist[0][bin_index])**2)/((math.sqrt((math.sqrt(real_hist[0][bin_index]))**2+(math.sqrt(fake_hist[0][bin_index]))**2))**2)
		if dof > 0:
			bdt_reduced_chi2 = bdt_chi2/dof
		else:
			bdt_reduced_chi2 = 1E8

		bdt_rchi2_list = np.append(bdt_rchi2_list, [[step, bdt_reduced_chi2]], axis=0)

		plt.plot(bdt_rchi2_list[:,0], bdt_rchi2_list[:,1])
		plt.xlabel('Training steps')
		plt.ylabel('rChi2')
		plt.yscale('log', nonposy='clip')
		plt.savefig('%s%s/test_output/bdt_rchi2.png'%(storage_location,file_number), bbox_inches='tight')
		plt.close('all')

		sum_overlap = 0

		for bin_index in range(0, 100):
			if real_hist[0][bin_index] < fake_hist[0][bin_index]:
				sum_overlap += real_hist[0][bin_index]
			elif fake_hist[0][bin_index] < real_hist[0][bin_index]:
				sum_overlap += fake_hist[0][bin_index]
			else:
				sum_overlap += fake_hist[0][bin_index]

		bdt_sum_overlap_list = np.append(bdt_sum_overlap_list, [[step, sum_overlap]], axis=0)

		plt.plot(bdt_sum_overlap_list[:,0], bdt_sum_overlap_list[:,1])
		plt.xlabel('Trianing steps')
		plt.ylabel('Overlap')
		plt.yscale('log', nonposy='clip')
		plt.savefig('%s%s/test_output/bdt_overlap.png'%(storage_location,file_number), bbox_inches='tight')
		plt.close('all')


		plot_data = np.concatenate((real_training_data, real_test_data))

		means_pos_x = np.load('%smeans_pos13_x_x.npy'%min_max_location)
		min_max_1_pos_x = np.load('%smin_max_pos13_x_x.npy'%min_max_location)
		min_max_2_pos_x = np.load('%smin_max_2_pos13_x_x.npy'%min_max_location)

		means_neg_x = np.load('%smeans_neg13_x_x.npy'%min_max_location)
		min_max_1_neg_x = np.load('%smin_max_neg13_x_x.npy'%min_max_location)
		min_max_2_neg_x = np.load('%smin_max_2_neg13_x_x.npy'%min_max_location)

		min_max_1_pos_0 = np.load('%smin_max_pos13_0_0.npy'%min_max_location)
		min_max_1_neg_0 = np.load('%smin_max_neg13_0_0.npy'%min_max_location)

		def normalize_6(input_array, min_max_1, min_max_2, means):
			for index in range(0, 2):
				for x in range(0, np.shape(input_array)[0]):
					input_array[x][index] = (((input_array[x][index]+0.97)/1.94)*(min_max_2[index][1] - min_max_2[index][0])+ min_max_2[index][0])
					input_array[x][index] = input_array[x][index] - means[index]
					if input_array[x][index] < 0:
						input_array[x][index] = -(input_array[x][index]**2)
					if input_array[x][index] > 0:
						input_array[x][index] = (input_array[x][index]**2)
					input_array[x][index] = input_array[x][index] + means[index]
					input_array[x][index] = (((input_array[x][index]+1)/2)*(min_max_1[index][1] - min_max_1[index][0])+ min_max_1[index][0])

			for index in range(2, 6):
				for x in range(0, np.shape(input_array)[0]):
					input_array[x][index] = (((input_array[x][index]+0.97)/1.94)*(min_max_1[index][1] - min_max_1[index][0])+ min_max_1[index][0])
			return input_array

		def normalize_4(input_array, min_max_1):
			for index in range(0, 4):
				for x in range(0, np.shape(input_array)[0]):
					input_array[x][index] = (((input_array[x][index]+0.97)/1.94)*(min_max_1[index][1] - min_max_1[index][0])+ min_max_1[index][0])
			return input_array

		def compare_cross_hists_2d_hist(index_1, index_2, axis_titles):
			plt.subplot(1,2,1)
			plt.title('Real Properties')
			plt.hist2d(plot_data[:,index_1],plot_data[:,index_2], bins = 100, norm=LogNorm(), range=[[-1,1],[-1,1]])
			plt.xlabel(axis_titles[index_1])
			plt.ylabel(axis_titles[index_2])
			plt.subplot(1,2,2)
			plt.title('Generated blur')
			plt.hist2d(images[:,0,index_1,0],images[:,0,index_2,0], bins = 100, norm=LogNorm(), range=[[-1,1],[-1,1]])
			plt.xlabel(axis_titles[index_1])
			plt.ylabel(axis_titles[index_2])
			plt.savefig('%s%s/test_output/compare_cross_blur_%d_%d_2D.png'%(storage_location,file_number,index_1, index_2), bbox_inches='tight')
			plt.close('all')

		if is_it_x == True:
			axis_titles = ['StartX', 'StartY', 'StartZ', 'Px', 'Py', 'Pz']
		else:
			axis_titles = ['StartZ', 'Px', 'Py', 'Pz']

		if is_it_x == True:
			for x in range(0,6):
				for y in range(x+1,6):
					compare_cross_hists_2d_hist(x,y, axis_titles)
		else:
			for x in range(0,4):
				for y in range(x+1,4):
					compare_cross_hists_2d_hist(x,y, axis_titles)

		if is_it_x == True:
			for index in range(0, 6):
				plt.hist([plot_data[:,index],images[:,0,index,0]], bins = 250,label=['real','gen'], histtype='step')
				plt.xlabel(axis_titles[index])
				plt.legend(loc='upper right')
				plt.savefig('%s%s/test_output/current_%d.png'%(storage_location,file_number,index), bbox_inches='tight')
				plt.savefig('%s%s/test_output/%d/current_%d.png'%(storage_location,file_number,index,step), bbox_inches='tight')
				plt.close('all')
		else:
			for index in range(0, 4):
				plt.hist([plot_data[:,index],images[:,0,index,0]], bins = 250,label=['real','gen'], histtype='step')
				plt.xlabel(axis_titles[index])
				plt.legend(loc='upper right')
				plt.savefig('%s%s/test_output/current_%d.png'%(storage_location,file_number,index), bbox_inches='tight')
				plt.savefig('%s%s/test_output/%d/current_%d.png'%(storage_location,file_number,index,step), bbox_inches='tight')
				plt.close('all')

		if save_name == 'neg13_0_0':
			images_squeeze = normalize_4(np.squeeze(images), min_max_1_neg_0)
			plot_data_squeeze = normalize_4(np.squeeze(plot_data), min_max_1_neg_0)
		if save_name == 'neg13_x_x':
			images_squeeze = normalize_6(np.squeeze(images), min_max_1_neg_x, min_max_2_neg_x, means_neg_x)
			plot_data_squeeze = normalize_6(np.squeeze(plot_data), min_max_1_neg_x, min_max_2_neg_x, means_neg_x)
		if save_name == 'pos13_0_0':
			images_squeeze = normalize_4(np.squeeze(images), min_max_1_pos_0)
			plot_data_squeeze = normalize_4(np.squeeze(plot_data), min_max_1_pos_0)
		if save_name == 'pos13_x_x':
			images_squeeze = normalize_6(np.squeeze(images), min_max_1_pos_x, min_max_2_pos_x, means_pos_x)
			plot_data_squeeze = normalize_6(np.squeeze(plot_data), min_max_1_pos_x, min_max_2_pos_x, means_pos_x)

		if dimensions == 6:
			pos_xy, z_real, mom_array = np.split(plot_data_squeeze,[2,3],axis=1)
		if dimensions == 4:
			z_real, mom_array = np.split(plot_data_squeeze,[1],axis=1)
		p_real = mom_array.sum(axis=1)
		p_x_real, p_y_real, p_z_real = np.split(mom_array, [1,2], axis=1)
		p_x_sq_real = np.multiply(p_x_real, p_x_real)
		p_y_sq_real = np.multiply(p_y_real, p_y_real)
		sqs_array_real = np.concatenate((p_x_sq_real,p_y_sq_real),axis=1)
		sum_sqs_real = sqs_array_real.sum(axis=1)
		p_t_real = np.sqrt(sum_sqs_real)
		p_real = np.expand_dims(p_real,1)
		p_t_real = np.expand_dims(p_t_real,1)

		real_mom = np.concatenate((z_real, p_x_real, p_y_real, p_real, p_t_real),axis=1)
		real_mom_train = real_mom[:50000]
		real_mom_test = real_mom[50000:]

		if dimensions == 6:
			pos_xy, z_fake, mom_array = np.split(images_squeeze,[2,3],axis=1)
		if dimensions == 4:
			z_fake, mom_array = np.split(images_squeeze,[1],axis=1)
		p_fake = mom_array.sum(axis=1)
		p_x_fake, p_y_fake, p_z_fake = np.split(mom_array, [1,2], axis=1)
		p_x_sq_fake = np.multiply(p_x_fake, p_x_fake)
		p_y_sq_fake = np.multiply(p_y_fake, p_y_fake)
		sqs_array_fake = np.concatenate((p_x_sq_fake,p_y_sq_fake),axis=1)
		sum_sqs_fake = sqs_array_fake.sum(axis=1)
		p_t_fake = np.sqrt(sum_sqs_fake)
		p_fake = np.expand_dims(p_fake,1)
		p_t_fake = np.expand_dims(p_t_fake,1)

		fake_mom = np.concatenate((z_fake, p_x_fake, p_y_fake, p_fake, p_t_fake),axis=1)
		fake_mom_train = fake_mom[:50000]
		fake_mom_test = fake_mom[50000:]

		plt.subplot(1,2,1)
		plt.hist2d(fake_mom_test[:,3], fake_mom_test[:,4],bins=100, norm=LogNorm(), range = [[np.amin(real_mom_test[:,3]),np.amax(real_mom_test[:,3])],[np.amin(real_mom_test[:,4]),np.amax(real_mom_test[:,4])]])
		plt.subplot(1,2,2)
		plt.hist2d(real_mom_test[:,3], real_mom_test[:,4],bins=100, norm=LogNorm(), range = [[np.amin(real_mom_test[:,3]),np.amax(real_mom_test[:,3])],[np.amin(real_mom_test[:,4]),np.amax(real_mom_test[:,4])]])
		plt.savefig('%s%s/test_output/p_pt/p_pt_%d.png'%(storage_location,file_number,step), bbox_inches='tight')
		plt.close('all')

		clf_mom = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)

		real_training_labels = np.ones(bdt_train_size)

		fake_training_labels = np.zeros(bdt_train_size)

		total_training_data = np.concatenate((real_mom_train, fake_mom_train))

		total_training_labels = np.concatenate((real_training_labels, fake_training_labels))

		clf_mom.fit(total_training_data, total_training_labels)

		out_real = clf_mom.predict_proba(real_mom_test)

		out_fake = clf_mom.predict_proba(fake_mom_test)

		plt.hist([out_real[:,1],out_fake[:,1]], bins = 100,label=['real','gen'], histtype='step')
		plt.xlabel('Output of BDT')
		plt.legend(loc='upper right')
		plt.savefig('%s%s/test_output/bdt_mom/BDT_out_%d.png'%(storage_location,file_number,step), bbox_inches='tight')
		plt.close('all')

		# Create histograms from the output of BDT.
		real_hist = np.histogram(out_real[:,1],bins=100,range=[0,1])
		fake_hist = np.histogram(out_fake[:,1],bins=100,range=[0,1])

		bdt_chi2_mom = 0
		dof_mom = 0

		for bin_index in range(0, 100):
			if real_hist[0][bin_index] > 9 and fake_hist[0][bin_index] > 9:
				dof_mom += 1
				bdt_chi2_mom += ((real_hist[0][bin_index] - fake_hist[0][bin_index])**2)/((math.sqrt((math.sqrt(real_hist[0][bin_index]))**2+(math.sqrt(fake_hist[0][bin_index]))**2))**2)
		if dof_mom > 0:
			bdt_reduced_chi2_mom = bdt_chi2_mom/dof_mom
		else:
			bdt_reduced_chi2_mom = 1E8

		print(bdt_rchi2_list_mom, np.shape(bdt_rchi2_list_mom))

		bdt_rchi2_list_mom = np.append(bdt_rchi2_list_mom, [[step, bdt_reduced_chi2_mom]], axis=0)

		plt.plot(bdt_rchi2_list_mom[:,0], bdt_rchi2_list_mom[:,1])
		plt.xlabel('Training steps')
		plt.ylabel('rChi2')
		plt.yscale('log', nonposy='clip')
		plt.savefig('%s%s/test_output/bdt_rchi2_mom.png'%(storage_location,file_number), bbox_inches='tight')
		plt.close('all')

		t1 = time.time()
		time_so_far = t1-t0
		with open("%s%s/test_output/rchi2_progress.txt"%(storage_location,file_number), "a") as myfile:
			myfile.write('%d, %.3f %.3f %.3f time: %.2f \n'%(step, bdt_reduced_chi2, bdt_reduced_chi2_mom, sum_overlap, time_so_far))

		return bdt_rchi2_list, bdt_sum_overlap_list, bdt_rchi2_list_mom

if __name__ == '__main__':
	
	gan = GAN()

	gan.train(dimensions)




