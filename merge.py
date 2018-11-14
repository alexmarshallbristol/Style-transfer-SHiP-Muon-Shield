import numpy as np
import glob

files = glob.glob('/eos/experiment/ship/user/amarshal/Ship_shield/out/*')

# files = files[:1]
total = np.empty((0,2,5))

for file in files:
	# here = np.load(file)
	# print(np.shape(here))
	current = np.load(file)
	total = np.append(total, current, axis=0)
	print(np.shape(total))

np.save('/eos/experiment/ship/user/amarshal/Ship_shield/merged_files',total)