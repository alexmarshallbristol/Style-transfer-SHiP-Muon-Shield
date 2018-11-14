import numpy as np

array_to_split = np.load('/eos/experiment/ship/user/amarshal/Ship_shield/merged_files.npy')

print(np.shape(array_to_split))

total_length = np.shape(array_to_split)[0]
single_length = np.floor(total_length/25)

print(single_length)

for x in range(0, 25):
	print(x)

	save_single = array_to_split[x*single_length:(x+1)*single_length]

	np.save('/eos/experiment/ship/user/amarshal/Ship_shield/training_files/geant_%d'%x, save_single)

print('Done...')
# split = np.array_split(array_to_split, 100)

# print(np.shape(split))
