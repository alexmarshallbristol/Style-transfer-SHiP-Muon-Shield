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

import argparse

from sklearn.ensemble import GradientBoostingClassifier


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

inital_state = Input(shape=(1,5))

H = generator([initial_state_w_noise,inital_state])

gan_output = discriminator(H)

GAN_stacked = Model(inputs=[initial_state_w_noise,inital_state], outputs=[gan_output])
GAN_stacked.compile(loss=_loss_generator, optimizer=optimizerD)
GAN_stacked.summary()



# Define arrays to save loss data ...

d_loss_list = g_loss_list = np.empty((0,2))

bdt_sum_overlap_list = np.empty((0,2))



# Define training variables and settings ...

epochs = int(1E30)

# Number of samples see per training step
batch = 50

# Number of samples from test_sample used in plots and BDT at every save_interval
test_batch = 25000

# Number of training steps between save_intervals
save_interval = 5000

# Number of training steps on a single training array before loading a new one
load_new_training_data = 10000

# Number of files (named 'geant_%d.npy') in output_location
number_of_files_in_folder = 25

#
running_on_choices = ['blue_crystal', 'deep_thought', 'craptop', 'blue_crystal_test']
runing_on = running_on_choices[0] 
#

# Choose approach for the random dimenion in generator input
approach_for_random_dimension_choices = ['narrow gaussian around 0', 'uniform'] 
approach_for_random_dimension = approach_for_random_dimension_choices[1]

#

# Define boundaries for pre and post processing - must match that in pre_process.py
boundaries_x_range = [-2500, 2500]
boundaries_y_range = [-2500, 2500]
boundaries_px_range = [-20,20]
boundaries_py_range = [-20,20]
boundaries_pz_range = [0, 400]

#

dimension_labels = ['x','y','p_x','p_y','p_z']

# Set directories
if runing_on == 'blue_crystal':
	training_data_location = '/mnt/storage/scratch/am13743/SHIP_SHIELD/training_files/'
	output_location = '/mnt/storage/scratch/am13743/SHIP_SHIELD/output/'

elif runing_on == 'deep_thought':
	training_data_location = 'training_files/'
	output_location = 'output/'

elif runing_on == 'craptop':
	training_data_location = '/Users/am13743/Desktop/style-transfer-GANs/data/training_files/'
	output_location = '/Users/am13743/Desktop/style-transfer-GANs/data/plots/'

elif runing_on == 'blue_crystal_test':
	training_data_location = '/mnt/storage/scratch/am13743/SHIP_SHIELD/training_files/'
	output_location = '/mnt/storage/scratch/am13743/SHIP_SHIELD/output_new/'
	save_interval = 5

'''
output_location should contain the following folders:

	- 0
	- 1
	- 2
	- 3
	- 4
	- BDT
	- angle
	- distance
	- momentum
'''

# Start training loop ...

for e in range(epochs):

	if e % load_new_training_data == 0:

		file_index = np.random.randint(0, number_of_files_in_folder)
	
		FairSHiP_sample = np.load('%sgeant_%d.npy'%(training_data_location,file_index))

		print('Loading new file (file %d), at training step'%file_index,e,'- new file has',np.shape(FairSHiP_sample)[0],'samples from FairSHiP.')

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

	if approach_for_random_dimension == 'narrow gaussian around 0':
		random_dimension = np.expand_dims(np.expand_dims(np.random.normal(0,0.01,batch),1),1)
	elif approach_for_random_dimension == 'uniform':
		random_dimension = np.expand_dims(np.expand_dims(np.random.rand(batch),1),1)

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

	if approach_for_random_dimension == 'narrow gaussian around 0':
		random_dimension = np.expand_dims(np.expand_dims(np.random.normal(0,0.01,batch),1),1)
	elif approach_for_random_dimension == 'uniform':
		random_dimension = np.expand_dims(np.expand_dims(np.random.rand(batch),1),1)

	g_training_w_noise = np.concatenate((g_training, random_dimension),axis=2) # Add dimension of random noise

	y_mislabled = np.ones((batch, 1))

	g_loss = GAN_stacked.train_on_batch([g_training_w_noise, g_training], y_mislabled)

	d_loss_list = np.append(d_loss_list, [[e,(d_loss_legit+d_loss_gen)/2]], axis=0)
	g_loss_list = np.append(g_loss_list, [[e, g_loss]], axis=0)

	if e % 1000 == 0 and e > 1: 
		print('Step:',e)

	if e % save_interval == 0 and e > 1: 

		print('Saving',e,'...')

		plt.subplot(1,2,1)
		plt.title('Discriminator loss')
		plt.plot(d_loss_list[:,0],d_loss_list[:,1])
		plt.subplot(1,2,2)
		plt.title('Generator loss')
		plt.plot(g_loss_list[:,0],g_loss_list[:,1])
		plt.savefig('%sloss.png'%output_location,bbox_inches='tight')
		plt.close('all')

		# Create test samples ... 

		random_indicies = np.random.choice(list_for_np_choice_test, size=test_batch, replace=False)

		sample_to_test = test_sample[random_indicies]

		if approach_for_random_dimension == 'narrow gaussian around 0':
			random_dimension = np.expand_dims(np.expand_dims(np.random.normal(0,0.01,test_batch),1),1)
		elif approach_for_random_dimension == 'uniform':
			random_dimension = np.expand_dims(np.expand_dims(np.random.rand(test_batch),1),1)

		sample_to_test_initial, throw_away = np.split(sample_to_test, [1], axis=1) # Remove the real final state information
		sample_to_test_initial_w_rand = np.concatenate((sample_to_test_initial, random_dimension),axis=2) # Add dimension of random noise

		synthetic_test_output = generator.predict([sample_to_test_initial_w_rand, sample_to_test_initial]) # Run initial muon parameters through G for a final state guess and initial state in shape (2,5)

		print('GEANT4 test sample shape:',np.shape(sample_to_test))
		print('GAN test sample shape:',np.shape(synthetic_test_output))


		def BDT(geant_output, gan_output):

			bdt_train_size = np.shape(geant_output)[0]

			geant_output_train = geant_output[:int(bdt_train_size/2)]
			geant_output_test = geant_output[int(bdt_train_size/2):]

			gan_output_train = gan_output[:int(bdt_train_size/2)]
			gan_output_test = gan_output[int(bdt_train_size/2):]

			clf_mom = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)

			real_training_labels = np.ones(int(bdt_train_size/2))

			fake_training_labels = np.zeros(int(bdt_train_size/2))

			total_training_data = np.concatenate((geant_output_train, gan_output_train))

			total_training_labels = np.concatenate((real_training_labels, fake_training_labels))

			clf_mom.fit(total_training_data, total_training_labels)

			out_real = clf_mom.predict_proba(geant_output_train)

			out_fake = clf_mom.predict_proba(gan_output_train)

			plt.hist([out_real[:,1],out_fake[:,1]], bins = 50, label=['GEANT4','GAN'], histtype='step')
			plt.title('Step %d'%e)
			plt.xlabel('Output of BDT')
			plt.legend(loc='upper right')
			plt.savefig('%sBDT/BDT_out_%d.png'%(output_location,e), bbox_inches='tight')
			plt.savefig('%scurrent_BDT_out.png'%output_location, bbox_inches='tight')
			plt.close('all')

			real_hist = np.histogram(out_real[:,1],bins=50,range=[0,1])
			fake_hist = np.histogram(out_fake[:,1],bins=50,range=[0,1])
		

			sum_overlap = 0

			for bin_index in range(0, 50):
				if real_hist[0][bin_index] < fake_hist[0][bin_index]:
					sum_overlap += real_hist[0][bin_index]
				elif fake_hist[0][bin_index] < real_hist[0][bin_index]:
					sum_overlap += fake_hist[0][bin_index]
				else:
					sum_overlap += fake_hist[0][bin_index]

			return sum_overlap



		overlap = BDT(sample_to_test[:,1],synthetic_test_output[:,1])

		bdt_sum_overlap_list = np.append(bdt_sum_overlap_list, [[e, overlap]], axis=0)

		plt.title('Overlap')
		plt.plot(bdt_sum_overlap_list[:,0],bdt_sum_overlap_list[:,1])
		plt.ylabel('Overlap (max possible is %d)'%int(np.shape(sample_to_test[:,1])[0]/2))
		plt.xlabel('Step')
		plt.savefig('%sBDT_overlap.png'%output_location,bbox_inches='tight')
		plt.close('all')

		# Post-process - recover real values

		def post_process(geant_array, gan_array):

			geant_array = (geant_array/2) + 0.5
			gan_array = (gan_array/2) + 0.5

			def post_process(input_array, minimum, full_range):

				input_array = input_array * full_range
				input_array += minimum

				return input_array

			for array in [geant_array, gan_array]:
				for i in range(0, 2):
					array[:,i,0] = post_process(array[:,i,0], boundaries_x_range[0], boundaries_x_range[1]-boundaries_x_range[0])
					array[:,i,1] = post_process(array[:,i,1], boundaries_y_range[0], boundaries_y_range[1]-boundaries_y_range[0])
					array[:,i,2] = post_process(array[:,i,2], boundaries_px_range[0], boundaries_px_range[1]-boundaries_px_range[0])
					array[:,i,3] = post_process(array[:,i,3], boundaries_py_range[0], boundaries_py_range[1]-boundaries_py_range[0])
					array[:,i,4] = post_process(array[:,i,4], boundaries_pz_range[0], boundaries_pz_range[1]-boundaries_pz_range[0])

			return geant_array, gan_array

		sample_to_test, synthetic_test_output = post_process(sample_to_test, synthetic_test_output)

		def plot_1d_hists(dim):

			plot_label = dimension_labels[dim]

			plt.figure(figsize=(8,8))

			plt.suptitle('step %d - Parameter - %s'%(e,plot_label))

			plt.subplot(3,2,1)
			#inital real
			plt.title('Inital unseen test input', fontsize='x-small')
			plt.hist(sample_to_test[:,0,dim], bins=50,range=[np.amin(sample_to_test[:,0,dim]),np.amax(sample_to_test[:,0,dim])])
			plt.subplot(3,2,2)
			#final real
			plt.title('GEANT4 output', fontsize='x-small')
			plt.hist(sample_to_test[:,1,dim], bins=50,range=[np.amin(sample_to_test[:,1,dim]),np.amax(sample_to_test[:,1,dim])])

			plt.subplot(3,2,3)
			#inital fake
			plt.title('Exact same test input', fontsize='x-small')
			plt.hist(synthetic_test_output[:,0,dim], bins=50,range=[np.amin(sample_to_test[:,0,dim]),np.amax(sample_to_test[:,0,dim])])
			plt.subplot(3,2,4)
			#final fake
			plt.title('GAN output - same range', fontsize='x-small')
			plt.hist(synthetic_test_output[:,1,dim], bins=50,range=[np.amin(sample_to_test[:,1,dim]),np.amax(sample_to_test[:,1,dim])])
			plt.subplot(3,2,5)
			#final fake - full range
			plt.title('GAN output - full range', fontsize='x-small')
			plt.hist(synthetic_test_output[:,1,dim], bins=50)

			plt.subplot(3,2,6)
			#final fake
			plt.title('Output overlays', fontsize='x-small')
			plt.hist([sample_to_test[:,1,dim],synthetic_test_output[:,1,dim]],histtype='step', bins=50,label=['GEANT4','GAN'],range=[np.amin(sample_to_test[:,1,dim]),np.amax(sample_to_test[:,1,dim])])
			plt.legend(loc='upper right')
			plt.tick_params(axis='y', which='both', labelsize=5)
			plt.tick_params(axis='x', which='both', labelsize=5)
			plt.savefig('%s1D_hist_%d.png'%(output_location,dim),bbox_inches='tight')
			plt.savefig('%s%d/1D_hist_%d.png'%(output_location,dim,e),bbox_inches='tight')

			plt.close('all')

		for first in range(0, 5):
			plot_1d_hists(first)

		def plot_2d_hists(dim_1, dim_2):

			plot_x_label = dimension_labels[dim_1]
			plot_y_label = dimension_labels[dim_2]

			plt.figure(figsize=(8,8))
			plt.suptitle('step %d - Parameters - %s against %s'%(e, plot_x_label, plot_y_label))
			plt.subplot(3,2,1)
			#inital real
			plt.title('Inital unseen test input', fontsize='x-small')
			plt.hist2d(sample_to_test[:,0,dim_1], sample_to_test[:,0,dim_2], bins=50,range=[[np.amin(sample_to_test[:,0,dim_1]),np.amax(sample_to_test[:,0,dim_1])],[np.amin(sample_to_test[:,0,dim_2]),np.amax(sample_to_test[:,0,dim_2])]],norm=LogNorm(), cmap=plt.cm.CMRmap)
			plt.xlabel('%s'%plot_x_label)
			plt.ylabel('%s'%plot_y_label)
			plt.subplot(3,2,2)
			#final real
			plt.title('GEANT4 output', fontsize='x-small')
			plt.hist2d(sample_to_test[:,1,dim_1], sample_to_test[:,1,dim_2], bins=50,range=[[np.amin(sample_to_test[:,1,dim_1]),np.amax(sample_to_test[:,1,dim_1])],[np.amin(sample_to_test[:,1,dim_2]),np.amax(sample_to_test[:,1,dim_2])]],norm=LogNorm(), cmap=plt.cm.CMRmap)
			plt.xlabel('%s'%plot_x_label)
			plt.ylabel('%s'%plot_y_label)

			plt.subplot(3,2,3)
			#inital fake
			plt.title('Exact same test input', fontsize='x-small')
			plt.hist2d(synthetic_test_output[:,0,dim_1], synthetic_test_output[:,0,dim_2], bins=50,range=[[np.amin(sample_to_test[:,0,dim_1]),np.amax(sample_to_test[:,0,dim_1])],[np.amin(sample_to_test[:,0,dim_2]),np.amax(sample_to_test[:,0,dim_2])]],norm=LogNorm(), cmap=plt.cm.CMRmap)
			plt.xlabel('%s'%plot_x_label)
			plt.ylabel('%s'%plot_y_label)

			plt.subplot(3,2,4)
			#final fake
			plt.title('GAN output - same range', fontsize='x-small')
			plt.hist2d(synthetic_test_output[:,1,dim_1], synthetic_test_output[:,1,dim_2], bins=50,range=[[np.amin(sample_to_test[:,1,dim_1]),np.amax(sample_to_test[:,1,dim_1])],[np.amin(sample_to_test[:,1,dim_2]),np.amax(sample_to_test[:,1,dim_2])]],norm=LogNorm(), cmap=plt.cm.CMRmap)
			plt.xlabel('%s'%plot_x_label)
			plt.ylabel('%s'%plot_y_label)

			plt.subplot(3,2,5)
			#final fake - full range
			plt.title('GAN output - full range', fontsize='x-small')
			plt.hist2d(synthetic_test_output[:,1,dim_1], synthetic_test_output[:,1,dim_2], bins=50,norm=LogNorm(), cmap=plt.cm.CMRmap)
			plt.xlabel('%s'%plot_x_label)
			plt.ylabel('%s'%plot_y_label)

			plt.tick_params(axis='y', which='both', labelsize=5)
			plt.tick_params(axis='x', which='both', labelsize=5)
			plt.savefig('%s2D_hist_%d_%d.png'%(output_location,dim_1,dim_2),bbox_inches='tight')

			plt.close('all')

		for first in range(0, 5):
			for second in range(first+1, 5):
				plot_2d_hists(first, second)


		# Plot deviated angle between input and output 3D momentum vectors

		def get_scattered_angle(array):

			array = np.swapaxes(array,0,2)

			input_mom = [array[2,0,:],array[3,0,:],array[4,0,:]]
			output_mom = [array[2,1,:],array[3,1,:],array[4,1,:]]

			input_mom = np.swapaxes(input_mom,0,1)
			output_mom = np.swapaxes(output_mom,0,1)

			dot_product = np.empty(0)
			product_of_magnitude = np.empty(0)

			# Can this loop be sped up with an element-wise dot product method - couldn't find one in numpy. 
			for x in range(0, int(np.shape(input_mom)[0])):
				dot_product = np.append(dot_product,np.dot(input_mom[x], output_mom[x]))
				mag_in = math.sqrt(input_mom[x][0]**2 + input_mom[x][1]**2 + input_mom[x][2]**2)
				mag_out = math.sqrt(output_mom[x][0]**2 + output_mom[x][1]**2 + output_mom[x][2]**2)
				product_of_magnitude = np.append(product_of_magnitude,mag_in*mag_out)

			cos_theta = np.divide(dot_product,product_of_magnitude)

			angle = np.arccos(cos_theta)

			return angle


		geant_angle = get_scattered_angle(sample_to_test)	
		gan_angle = get_scattered_angle(synthetic_test_output)	

		plt.figure(figsize=(8,8))
		plt.suptitle('step %d'%(e))
		plt.subplot(2,2,1)
		plt.title('Normal plot', fontsize='x-small')
		plt.hist([geant_angle,gan_angle],bins=50,histtype='step',label=['GEANT4','GAN'],range=[np.amin(geant_angle),np.amax(geant_angle)])
		plt.xlabel('Deviated angle (rads)')
		plt.legend(loc='upper right')
		plt.subplot(2,2,2)
		plt.title('Log plot', fontsize='x-small')
		plt.hist([geant_angle,gan_angle],bins=50,histtype='step',label=['GEANT4','GAN'],range=[np.amin(geant_angle),np.amax(geant_angle)])
		plt.yscale('log')
		plt.xlabel('Deviated angle (rads)')
		plt.legend(loc='upper right')

		# Now plot with full ranges
		plt.subplot(2,2,3)
		plt.title('Normal plot full range', fontsize='x-small')
		plt.hist([geant_angle,gan_angle],bins=50,histtype='step',label=['GEANT4','GAN'])
		plt.xlabel('Deviated angle (rads)')
		plt.legend(loc='upper right')
		plt.subplot(2,2,4)
		plt.title('Log plot full range', fontsize='x-small')
		plt.hist([geant_angle,gan_angle],bins=50,histtype='step',label=['GEANT4','GAN'])
		plt.yscale('log')
		plt.xlabel('Deviated angle (rads)')
		plt.legend(loc='upper right')

		plt.tick_params(axis='y', which='both', labelsize=5)
		plt.tick_params(axis='x', which='both', labelsize=5)
		plt.savefig('%sangle/angle_%d.png'%(output_location,e),bbox_inches='tight')
		plt.savefig('%scurrent_angle.png'%output_location,bbox_inches='tight')
		plt.close('all')


		# X - Y distance error 

		def get_x_y_error(geant_array,gan_array):

			geant_array = np.swapaxes(geant_array,0,1)[1]
			gan_array = np.swapaxes(gan_array,0,1)[1]

			difference = np.subtract(geant_array, gan_array)
			difference = np.swapaxes(difference,0,1)

			distance = np.sqrt(np.add(difference[0]**2,difference[1]**2))

			plt.figure(figsize=(8,8))
			plt.suptitle('step %d'%(e))

			plt.subplot(2,2,1)
			plt.title('normal plot', fontsize='x-small')
			plt.hist(distance,bins=50,histtype='step')
			plt.xlabel('Distance guess is out by (cm)')
			plt.subplot(2,2,2)
			plt.title('log plot', fontsize='x-small')
			plt.hist(distance,bins=50,histtype='step')
			plt.yscale('log')
			plt.xlabel('Distance guess is out by (cm)')

			plt.subplot(2,2,3)
			plt.title('normal plot', fontsize='x-small')
			plt.hist(distance,bins=50,histtype='step',range=[0,50])
			plt.xlabel('Distance guess is out by (cm)')
			plt.subplot(2,2,4)
			plt.title('log plot', fontsize='x-small')
			plt.hist(distance,bins=50,histtype='step',range=[0,50])
			plt.yscale('log')
			plt.xlabel('Distance guess is out by (cm)')

			plt.savefig('%scurrent_distance_out.png'%(output_location),bbox_inches='tight')
			plt.savefig('%sdistance/distance_out_%d.png'%(output_location,e),bbox_inches='tight')
			plt.close('all')

		get_x_y_error(sample_to_test,synthetic_test_output)

		# p pt plot

		def plot_p_against_pt(geant_array,gan_array):

			geant_array = np.swapaxes(geant_array,0,1)[1]
			gan_array = np.swapaxes(gan_array,0,1)[1]

			geant_array = np.swapaxes(geant_array,0,1)
			gan_array = np.swapaxes(gan_array,0,1)

			trans_geant = np.sqrt(np.add(geant_array[2]**2,geant_array[3]**2))
			trans_gan = np.sqrt(np.add(gan_array[2]**2,gan_array[3]**2))

			total_mom_geant = np.add(trans_geant, geant_array[4])
			total_mom_gan = np.add(trans_gan, gan_array[4])


			plt.figure(figsize=(8,8))
			plt.suptitle('step %d'%(e))

			plt.subplot(2,2,1)
			plt.title('p vs pt GEANT4', fontsize='x-small')
			plt.hist2d(total_mom_geant,trans_geant,bins=50,norm=LogNorm(), cmap=plt.cm.CMRmap)
			plt.xlabel('P (GeV)')
			plt.ylabel('P_t (GeV)')

			plt.subplot(2,2,2)
			plt.title('p vs pt GAN', fontsize='x-small')
			plt.hist2d(total_mom_gan,trans_gan,bins=50,norm=LogNorm(), cmap=plt.cm.CMRmap)
			plt.xlabel('P (GeV)')
			plt.ylabel('P_t (GeV)')

			plt.subplot(2,2,3)
			plt.title('p overlay', fontsize='x-small')
			plt.hist([total_mom_geant,total_mom_gan],bins=50,histtype='step',label=['GEANT4','GAN'])
			plt.xlabel('P (GeV)')
			plt.yscale('log')
			plt.legend(loc='upper right')

			plt.subplot(2,2,4)
			plt.title('p_t overlay', fontsize='x-small')
			plt.hist([trans_geant,trans_gan],bins=50,histtype='step',label=['GEANT4','GAN'])
			plt.xlabel('P_t (GeV)')
			plt.yscale('log')
			plt.legend(loc='upper right')

			plt.savefig('%smomentum/momentum_%d.png'%(output_location,e),bbox_inches='tight')
			plt.savefig('%scurrent_momentum.png'%output_location,bbox_inches='tight')
			plt.close('all')

		plot_p_against_pt(sample_to_test,synthetic_test_output)

		print('Saving complete...')






