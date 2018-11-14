import numpy as np
import matplotlib as mpl
mpl.use('TkAgg') 
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


simulated_array = np.load('simulated_array.npy')

print(np.shape(simulated_array))
# plt.hist2d(simulated_array)

# plt.hist2d(simulated_array[:,1,0],simulated_array[:,1,1],bins=50, norm=LogNorm())
# plt.savefig('test.png')

# quit()





print(np.shape(simulated_array))

# x_range = [-2500, 2500]
# y_range = [-2500, 2500]
# px_range = [-20,20]
# py_range = [-20,20]
# pz_range = [0, 400]


# simulated_array = np.reshape(simulated_array,(2,np.shape(simulated_array)[0],5))
simulated_array = np.swapaxes(simulated_array,0,1)

print(np.shape(simulated_array))
print(np.shape(simulated_array[0]))

print(simulated_array[0])

# x_i, rest = np.split(simulated_array[0],[0],axis=1)

# improve all this with element wise division?


x_i = simulated_array[0,:,0]
y_i = simulated_array[0,:,1]
px_i = simulated_array[0,:,2]
py_i = simulated_array[0,:,3]
pz_i = simulated_array[0,:,4]

def process(input_array, minimum, full_range):
	input_array += minimum * (-1)
	input_array = input_array/full_range

	return input_array

x_i = process(x_i, -2500, 5000)
y_i = process(y_i, -2500, 5000)
px_i = process(px_i, -20, 40)
py_i = process(py_i, -20, 40)
pz_i = process(pz_i, 0, 400)


inital = [x_i, y_i, px_i, py_i, pz_i]

x_f = simulated_array[1,:,0]
y_f = simulated_array[1,:,1]
px_f = simulated_array[1,:,2]
py_f = simulated_array[1,:,3]
pz_f = simulated_array[1,:,4]

x_f = process(x_f, -2500, 5000)
y_f = process(y_f, -2500, 5000)
px_f = process(px_f, -20, 40)
py_f = process(py_f, -20, 40)
pz_f = process(pz_f, 0, 400)

final = [x_f, y_f, px_f, py_f, pz_f]

simulated_array = [inital, final]

print(np.shape(simulated_array))

# simulated_array = np.reshape(simulated_array,(np.shape(simulated_array)[2],2,5))
simulated_array = np.swapaxes(simulated_array,1,0)
simulated_array = np.swapaxes(simulated_array,1,2)
simulated_array = np.swapaxes(simulated_array,1,0)
simulated_array = np.swapaxes(simulated_array,1,2)

simulated_array = (simulated_array - 0.5) * 2

print(np.shape(simulated_array))

np.save('simulated_array_pre_processed',simulated_array)











