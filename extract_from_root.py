import ROOT 
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import math
from matplotlib.colors import LogNorm

# Open file 
# f = ROOT.TFile.Open("example.root")

f = ROOT.TFile.Open("ship.conical.MuonBack-TGeant4.root")
# Get ya tree
tree = f.Get("cbmsim")

# Get length of tree
N = tree.GetEntries()

in_array = np.empty((0,5))
out_array = np.empty((0,5))

print(int(N), 'GetEntries')
# For every entry in tree
for i in xrange(N):

	# if i % 1000 == 0: print(i, '/',np.shape(save_array)[0])

	# Get entry
	tree.GetEntry(i)

	# Assign python labels to your branches
	mctracks = tree.MCTrack
	muonpoints = tree.muonPoint

	check = [False,False]

	for m in muonpoints:
		
		if m.GetDetectorID() == 1085 and m.PdgCode() == 13:
			check[0] = True
		if m.GetDetectorID() == 1086 and m.PdgCode() == 13:
			check[1] = True

	if check[0] == True and check[1] == True:
		for m in muonpoints:
			if m.GetDetectorID() == 1085 and m.PdgCode() == 13:
				in_array = np.append(in_array, [[m.GetX(), m.GetY(), m.GetPx(), m.GetPy(), m.GetPz()]],axis=0)
			if m.GetDetectorID() == 1086 and m.PdgCode() == 13:
				out_array = np.append(out_array, [[m.GetX(), m.GetY(), m.GetPx(), m.GetPy(), m.GetPz()]],axis=0)




		# if m.GetZ() == -6001:
			# out_array = np.append(out_array, [[m.GetX(), m.GetY(), m.GetPx(), m.GetPy(), m.GetPz()]],axis=0)

			# for t in mctracks:
			# 	if t.GetStartZ() == -6500:
			# 		in_array = np.append(in_array, [[t.GetStartX(), t.GetStartY(), t.GetPx(), t.GetPy(), t.GetPz()]],axis=0)
			# 		break
			# break



# quit()

# to_delete = np.where(in_array[:,4] > 100)
# in_array = np.delete(in_array, to_delete, axis=0)
# out_array = np.delete(out_array, to_delete, axis=0)


simulated_array = [in_array, out_array]
print(np.shape(simulated_array))

# simulated_array = np.reshape(simulated_array,(np.shape(simulated_array)[1],2,5))
simulated_array = np.swapaxes(simulated_array,0,1)
print(np.shape(simulated_array))


np.save('simulated_array',simulated_array)

# plt.figure(figsize=(8,4))
# plt.subplot(1,2,1)
# plt.hist2d(in_array[:,0],in_array[:,1],range=[[-1000,1000],[-1000,1000]],bins=50,norm=LogNorm())
# plt.subplot(1,2,2)
# plt.hist2d(out_array[:,0],out_array[:,1],range=[[-1000,1000],[-1000,1000]],bins=50,norm=LogNorm())
# plt.savefig('test.png')
# plt.close('all')

# plt.figure(figsize=(8,4))
# plt.subplot(1,2,1)
# plt.hist2d(simulated_array[:,0,0],simulated_array[:,0,1],range=[[-1000,1000],[-1000,1000]],bins=50,norm=LogNorm())
# plt.subplot(1,2,2)
# plt.hist2d(simulated_array[:,1,0],simulated_array[:,1,1],range=[[-1000,1000],[-1000,1000]],bins=50,norm=LogNorm())
# plt.savefig('test.png')
# plt.close('all')








