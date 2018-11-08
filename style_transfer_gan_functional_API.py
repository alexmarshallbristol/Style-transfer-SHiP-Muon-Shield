
'''

G goes 6 -> 5

D goes 5 -> 1

'''

# https://stackoverflow.com/questions/47947578/multi-input-models-using-keras-model-api



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

inital_state = Input(shape=(1,5))

H = Dense(256)(initial_state_w_noise)
H = LeakyReLU(alpha=0.2)(H)
H = BatchNormalization(momentum=0.8)(H)

H = Dense(512)(H)
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

H = Dense(256)(H)
H = LeakyReLU(alpha=0.2)(H)

H = Dense(512)(H)
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

gan_input = Input(shape=[100])

initial_state_w_noise = Input(shape=(1,6))

inital_state = Input(shape=(1,5))

H = generator([initial_state_w_noise,inital_state])

gan_output = discriminator(H)

GAN_stacked = Model(inputs=[initial_state_w_noise,inital_state], outputs=[gan_output])
GAN_stacked.compile(loss=_loss_generator, optimizer=optimizerD)
GAN_stacked.summary()



# Define arrays to save loss data ...

d_loss_list = g_loss_list = np.empty(0)


# Define training variables ...

epochs = 10000
batch = 50
save_interval = 100


# Load data ...

FairSHiP_sample = np.random.rand(1000,2,5) # In reality will use np.load() here 
print(np.shape(FairSHiP_sample)[0],'samples from FairSHiP.')
split = [0.7,0.3]
print('Split:',split)
split_index = int(split[0]*np.shape(FairSHiP_sample)[0])
list_for_np_choice = np.arange(np.shape(FairSHiP_sample)[0]) 
random_indicies = np.random.choice(list_for_np_choice, size=np.shape(FairSHiP_sample)[0], replace=False)
training_sample = FairSHiP_sample[:split_index]
test_sample = FairSHiP_sample[split_index:]

list_for_np_choice = np.arange(np.shape(training_sample)[0]) 


# Start training loop ...

for e in range(epochs):

	random_indicies = np.random.choice(list_for_np_choice, size=(3,batch), replace=False)

	# Prepare training samples for D ...

	d_real_training = training_sample[random_indicies[0]]
	d_fake_training = training_sample[random_indicies[1]]

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
	random_dimension = np.expand_dims(np.expand_dims(np.random.rand(batch),1),1)
	g_training_w_noise = np.concatenate((g_training, random_dimension),axis=2) # Add dimension of random noise



	y_mislabled = np.ones((batch, 1))

	g_loss = GAN_stacked.train_on_batch([g_training_w_noise, g_training], y_mislabled)



