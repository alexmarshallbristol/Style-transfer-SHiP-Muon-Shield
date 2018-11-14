import numpy as np
import time
import ROOT

z_start = -6500

x_lim = 100
y_lim = 100
px_lim = 10
py_lim = 10
pz_lim = [10,400]


for file in range(0, 1):

	num_to_generate_full = 100000

	x_column = np.multiply(np.subtract(np.random.rand(num_to_generate_full),np.ones(num_to_generate_full)*0.5)*2,np.ones(num_to_generate_full)*x_lim)
	y_column = np.multiply(np.subtract(np.random.rand(num_to_generate_full),np.ones(num_to_generate_full)*0.5)*2,np.ones(num_to_generate_full)*y_lim)
	px_column = np.multiply(np.subtract(np.random.rand(num_to_generate_full),np.ones(num_to_generate_full)*0.5)*2,np.ones(num_to_generate_full)*px_lim)
	py_column = np.multiply(np.subtract(np.random.rand(num_to_generate_full),np.ones(num_to_generate_full)*0.5)*2,np.ones(num_to_generate_full)*py_lim)
	pz_column = np.add(np.random.rand(num_to_generate_full)*(pz_lim[1]-pz_lim[0]),np.ones(num_to_generate_full)*pz_lim[0])

	generated_sample_full = [x_column,y_column,px_column,py_column,pz_column]

	num_to_generate_low = 100000

	x_column = np.multiply(np.subtract(np.random.rand(num_to_generate_low),np.ones(num_to_generate_low)*0.5)*2,np.ones(num_to_generate_low)*x_lim)
	y_column = np.multiply(np.subtract(np.random.rand(num_to_generate_low),np.ones(num_to_generate_low)*0.5)*2,np.ones(num_to_generate_low)*y_lim)
	px_column = np.multiply(np.subtract(np.random.rand(num_to_generate_low),np.ones(num_to_generate_low)*0.5)*2,np.ones(num_to_generate_low)*px_lim/2)
	py_column = np.multiply(np.subtract(np.random.rand(num_to_generate_low),np.ones(num_to_generate_low)*0.5)*2,np.ones(num_to_generate_low)*py_lim/2)
	pz_column = np.add(np.random.rand(num_to_generate_low)*((pz_lim[1]-pz_lim[0])/2),np.ones(num_to_generate_low)*(pz_lim[0]/2))

	generated_sample_low = [x_column,y_column,px_column,py_column,pz_column]

	# generated_sample = np.concatenate((generated_sample_full, generated_sample_low),axis=1)

	generated_sample = generated_sample_full





	print('Generated shape:',np.shape(generated_sample))

	# quit()


	print(file,'Opening ROOT file...')

	# file_ouput = '/eos/experiment/ship/user/amarshal/Ship_shield/uniform_muons/uniform_muons_%d.root'%file
	file_ouput = 'uniform_muons.root'
	f = ROOT.TFile(file_ouput, "recreate")
	t = ROOT.TTree("pythia8-Geant4", "pythia8-Geant4")

	float64 = 'd'
	event_id = np.zeros(1, dtype=float64)
	id_ = np.zeros(1, dtype=float64)
	parentid = np.zeros(1, dtype=float64)
	pythiaid = np.zeros(1, dtype=float64)
	ecut = np.zeros(1, dtype=float64)
	w = np.zeros(1, dtype=float64)
	vx = np.zeros(1, dtype=float64)
	vy = np.zeros(1, dtype=float64)
	vz = np.zeros(1, dtype=float64)
	px = np.zeros(1, dtype=float64)
	py = np.zeros(1, dtype=float64)
	pz = np.zeros(1, dtype=float64)
	release_time = np.zeros(1, dtype=float64)
	mother_id = np.zeros(1, dtype=float64)
	process_id = np.zeros(1, dtype=float64)

	t.Branch('event_id', event_id, 'event_id/D')
	t.Branch('id', id_, 'id/D')
	t.Branch('parentid', parentid, 'parentid/D')
	t.Branch('pythiaid', pythiaid, 'pythiaid/D')
	t.Branch('ecut', ecut, 'ecut/D')
	t.Branch('w', w, 'w/D')#relative weight
	t.Branch('x', vx, 'vx/D')#pos
	t.Branch('y', vy, 'vy/D')
	t.Branch('z', vz, 'vz/D')
	t.Branch('px', px, 'px/D')
	t.Branch('py', py, 'py/D')
	t.Branch('pz', pz, 'pz/D')
	t.Branch('release_time', release_time, 'release_time/D')
	t.Branch('mother_id', mother_id, 'mother_id/D')
	t.Branch('process_id', process_id, 'process_id/D')

	# t.Branch('event_id', event_id, 'event_id/F')
	# t.Branch('id', id_, 'id/F')
	# t.Branch('parentid', parentid, 'parentid/F')
	# t.Branch('pythiaid', pythiaid, 'pythiaid/F')
	# t.Branch('ecut', ecut, 'ecut/F')
	# t.Branch('w', w, 'w/F')#relative weight
	# t.Branch('x', vx, 'vx/F')#pos
	# t.Branch('y', vy, 'vy/F')
	# t.Branch('z', vz, 'vz/F')
	# t.Branch('px', px, 'px/F')
	# t.Branch('py', py, 'py/F')
	# t.Branch('pz', pz, 'pz/F')
	# t.Branch('release_time', release_time, 'release_time/F')
	# t.Branch('mother_id', mother_id, 'mother_id/F')
	# t.Branch('process_id', process_id, 'process_id/F')

	def save(event_id_in, pdg_in, x_in, y_in, z_in, px_in, py_in, pz_in, mother_id_in, process_id_in):
				event_id[0] = event_id_in
				id_[0] = 0
				parentid[0] = 0
				pythiaid[0] = pdg_in
				ecut[0] = 0.00001
				w[0] = 1
				vx[0] = x_in
				vy[0] = y_in
				# vz[0] = z_in
				vz[0] = -6002
				# px[0] = px_in
				# py[0] = py_in
				px[0] = 0
				py[0] = 0
				pz[0] = pz_in
				release_time[0] = 0
				mother_id[0] = mother_id_in
				process_id[0] = process_id_in
				t.Fill()


	for i in range(0, np.shape(generated_sample)[1]):
		save(1,13,generated_sample[0][i],generated_sample[1][i],z_start,generated_sample[2][i],generated_sample[3][i],generated_sample[4][i],99,99)

	print(np.shape(generated_sample))

	print('Writing file...')

	f.Write()
	f.Close()

	print('Done...')












