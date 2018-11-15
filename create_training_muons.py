import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.stats as stats

# z_start = -6500

x_lim = 100
y_lim = 100
px_lim = 10
py_lim = 10
pz_lim = [10,400]



# num_to_generate_full = 100000

# # Uniform x and y
# x_column = np.multiply(np.subtract(np.random.rand(num_to_generate_full),np.ones(num_to_generate_full)*0.5)*2,np.ones(num_to_generate_full)*x_lim)
# y_column = np.multiply(np.subtract(np.random.rand(num_to_generate_full),np.ones(num_to_generate_full)*0.5)*2,np.ones(num_to_generate_full)*y_lim)






# # px_column = np.multiply(np.subtract(np.random.rand(num_to_generate_full),np.ones(num_to_generate_full)*0.5)*2,np.ones(num_to_generate_full)*px_lim)
# # py_column = np.multiply(np.subtract(np.random.rand(num_to_generate_full),np.ones(num_to_generate_full)*0.5)*2,np.ones(num_to_generate_full)*py_lim)

# px_column = np.random.normal(loc=0,scale=10,size=num_to_generate_full)
# py_column = np.random.normal(loc=0,scale=10,size=num_to_generate_full)


# pz_column = np.add(np.random.rand(num_to_generate_full)*(pz_lim[1]-pz_lim[0]),np.ones(num_to_generate_full)*pz_lim[0])

# generated_sample_full = [x_column,y_column,px_column,py_column,pz_column]



data = np.load('/Users/am13743/Desktop/style-transfer-GANs/data/huge_generation_456.npy')

print(np.shape(data))

# x_column = data[:,1]
# y_column = data[:,2]
x_column = np.multiply(np.subtract(np.random.rand(np.shape(data)[0]),np.ones(np.shape(data)[0])*0.5)*2,np.ones(np.shape(data)[0])*x_lim)
y_column = np.multiply(np.subtract(np.random.rand(np.shape(data)[0]),np.ones(np.shape(data)[0])*0.5)*2,np.ones(np.shape(data)[0])*y_lim)

px_column = data[:,4]
py_column = data[:,5]
pz_column = data[:,6]


lower, upper = 0, 5
mu, sigma = 1.5, 1
X_pt = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
X_pt = X_pt.rvs((2,np.shape(data)[0]))

lower, upper = 0.25, 1.2
mu, sigma = 1.1, 1
X_pz = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
X_pz = X_pz.rvs(np.shape(data)[0])





plt.figure(figsize=(12,8))

generated_sample_full = [x_column,y_column,px_column,py_column,pz_column]

p_t = np.sqrt(np.add(generated_sample_full[2]**2,generated_sample_full[3]**2))
p = p_t + generated_sample_full[4]

plt.subplot(2,3,2)
plt.title('P vs P_t from GAN')
plt.hist2d(p,p_t,bins=50,norm=LogNorm(),range=[[0,400],[0,20]])
plt.xlabel('p')
plt.xlabel('p_t')

plt.subplot(2,3,4)
plt.title('px and py pre blur')
plt.hist([generated_sample_full[2],generated_sample_full[3]],label=['px','py'],bins=50,histtype='step',range=[-5,5])
plt.xlabel('px / py')
plt.legend(loc='upper right')

px_column = px_column * X_pt[0]
py_column = py_column * X_pt[1]
pz_column = pz_column * X_pz


generated_sample_full = [x_column,y_column,px_column,py_column,pz_column]


p_t = np.sqrt(np.add(generated_sample_full[2]**2,generated_sample_full[3]**2))
p = p_t + generated_sample_full[4]


plt.subplot(2,3,1)
plt.title('uniform x vs y distribution')
plt.hist2d(generated_sample_full[0],generated_sample_full[1],bins=50,norm=LogNorm())
plt.xlabel('x')
plt.xlabel('y')

plt.subplot(2,3,3)
plt.title('Widened P vs P_t')
plt.hist2d(p,p_t,bins=50,norm=LogNorm(),range=[[0,400],[0,20]])
plt.xlabel('p')
plt.xlabel('p_t')

plt.subplot(2,3,5)
plt.title('px and py post blur')
plt.hist([generated_sample_full[2],generated_sample_full[3]],label=['px','py'],bins=50,histtype='step',range=[-5,5])
plt.xlabel('px / py')
plt.legend(loc='upper right')


# plt.show()
# plt.close('all')
plt.savefig('example_training_blur_to_GAN_muons.png', bbox_inches='tight')



