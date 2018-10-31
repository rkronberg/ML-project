import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from scipy.special import comb


## This part of the code reads the raw data (.xyz files) and returns the central quantities stored in arrays

def preprocess(datasize):

	# Selects all molecules with 7 or fewer non-H atoms (3963) and (datasize - 3963) molecules with 8 non-H atoms at random.
	# This compensates the underrepresentation of small molecules (molecules with 9 non-H atoms are excluded)

	ind = np.concatenate((np.arange(1,3964),np.random.randint(3964,21989,size=datasize-3963)))

	# Initialize the variables as empty lists
	# natoms = number of atoms in a given molecule
	# nonHatoms = number of non-H atoms in a given molecule 21989
	# Ea = Atomization energy (Ha)
	# mu = Dipole moment (Debye
	# alpha = Isotropic polarizability (bohr^3)
	# atomlist = list of the atoms constituting a given molecule (e.g. ['C','H','H','H'] for methane)
	# coords = xyz coordinates of each atom in a given molecule
	# charges = Partial charges from Mulliken population analysis (e)

	natoms,nonHatoms,Ea,charges,alpha,mu,gap,atomlist,coords=[],[],[],[],[],[],[],[],[]

	atomref=[-0.500273,-37.846772,-54.583861,-75.064579,-99.718730]     # Energies (Ha) of single atoms [H,C,N,O,F]
	atoms=['H','C','N','O','F']

	# Loop over all selected indices (molecules)

	for i in ind:
		xyz,elemtype,mulliken,nnonH=[],[],[],0      # Initialize list that will contain coordinates and element types of ith molecule
		i = str(i).zfill(6)         # This pads the index with zeros so that all contain 6 digits (e.g. index 41 -> 000041)
		
		# Define the path to the .xyz file of ith molecule. Here it is assumed that the dataset is stored in a 
		# subdirectory "xyz" within the one containing machine.py

		fpath = os.path.join('xyz',"dsgdb9nsd_%s.xyz" % i)      # xyz/*.xyz

		# Open the file and loop over the lines

		with open(fpath) as f:
			for j, line in enumerate(f):
				if j == 0:
					na = int(line)      # Number of atoms in molecule
					natoms.append(na)
				elif j == 1:
					E = float(line.split()[12])                     # Properties written on second line. Atomization energy,
					mu.append(float(line.split()[5]))               # Dipole moment,
					alpha.append(float(line.split()[6])*0.14818)    # Polarizability
					gap.append(float(line.split()[9])*27.21139)     # HOMO-LUMO gap
				elif 2 <= j <= na+1:
					parts = line.split()                    # Lines 2 -> na+1 contains element types, coordinates and charges
					elemtype.append(parts[0])               # Index 0 = element type, 1 = x, 2 = y, 3 = z
					mulliken.append(parts[4])               # Partial charge on atom
					E = E - atomref[atoms.index(parts[0])]  # Subtract energy of isolated atom from total energy
					if parts[0] != 'H':
						nnonH += 1
					xyz.append(np.array([float(parts[1]),float(parts[2]),float(parts[3])]))

		Ea.append(-E*27.21139)
		atomlist.append(elemtype)
		coords.append(xyz)
		nonHatoms.append(nnonH)
		charges.append(mulliken)

	# Return all lists in the form of numpy arrays

	return np.array(natoms),np.array(Ea),np.array(mu),np.array(charges),np.array(alpha),np.array(gap), \
		np.array(atomlist),np.array(coords),np.array(nonHatoms)


## Implement the MBTR descriptor here! Azeema?

def mbtr(mbtr_input):



	return mbtr_output


## The BoB descriptor

def bob(atomlist,coords):

	atoms = ['H','C','O','N','F']
	Z = [1,6,8,7,9]     
	bob_output = []
	dim = int(comb(18,2))			# 18 H atoms in octane -> comb(18,2) H-H pairs (max. size of a bond vector in a bag of bonds)

	for i in range(len(atomlist)):
		bag = {'HH': dim*[0],'HC': dim*[0],'HO': dim*[0],'HN': dim*[0],'HF': dim*[0],'CC': dim*[0],'CO': dim*[0],
		'CN': dim*[0],'CF': dim*[0],'OO': dim*[0],'ON': dim*[0],'OF': dim*[0],'NN': dim*[0],'NF': dim*[0],'FF': dim*[0]}	# General form of a bag of bonds
		Bvec = np.array([])
		for j in range(len(atomlist[i])):
			for k in range(len(atomlist[i])):
				if j > k:
					try:
						bag[atomlist[i][j]+atomlist[i][k]].insert(0,Z[atoms.index(atomlist[i][j])]*Z[atoms.index(atomlist[i][k])]/np.linalg.norm(coords[i][j]-coords[i][k]))
						del bag[atomlist[i][j]+atomlist[i][k]][-1]
					except KeyError:
						bag[atomlist[i][k]+atomlist[i][j]].insert(0,Z[atoms.index(atomlist[i][j])]*Z[atoms.index(atomlist[i][k])]/np.linalg.norm(coords[i][j]-coords[i][k]))
						del bag[atomlist[i][k]+atomlist[i][j]][-1]		# Avoid KeyError raised by "wrong" order of atoms in a bond (e.g. 'CH' -> 'HC')
		
		for pair in bag:
			Bvec = np.concatenate((Bvec,np.array(bag[pair])))

		bob_output.append(Bvec)

	return bob_output

## The following function takes the number of atoms in each molecule, the atom types and corresponding coordinates 
## and returns an array of corresponding Coulomb matrices

def coulomb(natoms,atomlist,coords):

	dim = natoms.max()                      # Specify the dimensions of the Coulomb matrices based on the largest molecule
	atoms = ['C','H','O','N','F']           # List of all possible atom types
	Z = [6,1,8,7,9]                         # The corresponding nuclear charges
	M = np.zeros((len(natoms),dim,dim))     # Initialize an array of all Coulomb matrices
	Mvec = []

	for i in range(len(natoms)):                # Loop over all molecules
		for j in range(len(atomlist[i])):       # Loop over all atom pairs (j,k) in molecule i
			for k in range(len(atomlist[i])):
				if j == k:
					M[i][j][k] = 0.5*Z[atoms.index(atomlist[i][j])]**2.4
				else:
					M[i][j][k] = Z[atoms.index(atomlist[i][j])]*Z[atoms.index(atomlist[i][k])]/np.linalg.norm(coords[i][j]-coords[i][k])
		
		# Sort Coulomb matrix according to descending row norm

		indexlist = np.argsort(-np.linalg.norm(M[i],axis=1))    # Get the indices in the sorted order
		M[i] = M[i][indexlist]                                  # Rearrange the matrix
		Mvec.append(M[i][np.tril_indices(dim,k=0)])             # Convert the lower triangular matrix into a vector and append 
																# to a list of Coulomb 'vectors' 

	return Mvec

## Do the grid search (if optimal hyperparameters are not known), then training and prediction using KRR
## If doing grid search for optimal parameters use small training set size, like 1k (takes forever otherwise)

def krr(x,y,nonHatoms):
	
	## Optimal hyperparameters for CM + Laplacian kernel
	# Ea: 				alpha 1e-11, gamma 1e-4
	# polarizability: 	alpha 1e-3, gamma 1e-4
	# HOMO-LUMO gap: 	alpha 1e-2, gamma 1e-4
	# Dipole moment: 	alpha 1e-1, gamma 1e-3

	## Optimal hyperparameters for BoB + Laplacian kernel
	# Ea: 				alpha 1e-11, gamma 1e-5
	# polarizability: 	alpha 1e-3, gamma 1e-4
	# HOMO-LUMO gap: 	alpha 1e-3, gamma 1e-4
	# Dipole moment: 	alpha 1e-1, gamma 1e-3

	## Optimal hyperparameters for MBTR + Gaussian kernel
	# Ea:
	# polarizability:
	# HOMO-LUMO gap:
	# Dipole moment:

	inp4 = input('Do grid search for optimal hyperparameters? [True/False]\n')

	if inp4 == True:

		x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.9,stratify=nonHatoms)
		kr = GridSearchCV(KernelRidge(kernel='laplacian'),cv=5,param_grid={"alpha": np.logspace(-11,0,12),"gamma": np.logspace(-11,0,12)})
		kr.fit(x_train,y_train)
		print(kr.best_params_)

	elif inp4 == False:

		inp5 = raw_input('Provide kernel and hyperparameters. [kernel alpha gamma]\n').split()

		x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,stratify=nonHatoms)
		kr = KernelRidge(kernel=inp5[0],alpha=float(inp5[1]),gamma=float(inp5[2]))
		kr.fit(x_train,y_train)
	
	y_pred = kr.predict(x_test)
	mae = MAE(y_test,y_pred)
	rmse = np.sqrt(MSE(y_test,y_pred))

	# Print mean absolute error and root mean squared error

	print('Mean absolute error: ',MAE(y_test,y_pred),'Root mean squared error: ',np.sqrt(MSE(y_test,y_pred)))

	return y_pred,y_test

def learning_curve(x,y,nonHatoms):

	# Do training with different sample sizes and see how the MAE behaves (learning curve)
	inp5 = raw_input('Provide kernel and hyperparameters. [kernel alpha gamma]\n').split()

	mae,rmse=[],[]
	sample_sizes = [50,200,1000,3000,9000]
	kr = KernelRidge(kernel=inp5[0],alpha=float(inp5[1]),gamma=float(inp5[2]))

	for i in sample_sizes:

		x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=1-float(i)/len(y),stratify=nonHatoms)

		kr.fit(x_train,y_train)
		y_pred = kr.predict(x_test)
		mae.append(MAE(y_test,y_pred))
		rmse.append(np.sqrt(MSE(y_test,y_pred)))

	return y_pred,y_test,mae,rmse,sample_sizes

## The main routine and plotting

def main():

	# Just some plot settings
	plt.ion()
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif', size=14)
	plt.rc('xtick', direction='in')

	# Preprocess data
	datasize=10000
	natoms,Ea,mu,charges,alpha,gap,atomlist,coords,nonHatoms = preprocess(datasize)

	inp1 = raw_input('Which descriptor? [CM/BoB/MBTR]\n')

	if inp1 == 'CM':

		descriptor = coulomb(natoms,atomlist,coords)
	
	elif inp1 == 'BoB':

		descriptor = bob(atomlist,coords)

	elif inp1 == 'MBTR':

		#descriptor = mbtr(mbtr_input)
		print('Not yet implemented.')

	inp2 = raw_input('Which property? [Ea/gap/alpha/mu]\n')

	plt.figure()

	if inp2 == 'Ea':

		prop = Ea
		plt.title(r'Atomization energy (eV)')
		plt.xlabel(r'$\Delta_\mathrm{at}E^\mathrm{DFT}$ (eV)')
		plt.ylabel(r'$\Delta_\mathrm{at}E^\mathrm{KRR}$ (eV)')

	elif inp2 == 'gap':

		prop = gap
		plt.title(r'HOMO-LUMO gap (eV)')
		plt.xlabel(r'$\Delta\varepsilon^\mathrm{DFT}$ (eV)')
		plt.ylabel(r'$\Delta\varepsilon^\mathrm{KRR}$ (eV)')

	elif inp2 == 'alpha':

		prop = alpha
		plt.title(r'Isotropic polarizability (\r{A}$^3$)')
		plt.xlabel(r'$\alpha^\mathrm{DFT}$ (\r{A}$^3$)')
		plt.ylabel(r'$\alpha^\mathrm{KRR}$ (\r{A}$^3$)')

	elif inp2 == 'mu':

		prop = mu
		plt.title(r'Dipole moment (D)')
		plt.xlabel(r'$\mu^\mathrm{DFT}$ (D)')
		plt.ylabel(r'$\mu^\mathrm{KRR}$ (D)')

	inp3 = input('Plot learning curve? [True/False]\n')

	if inp3 == True:

		# Train
		y_pred,y_test,mae,rmse,sample_sizes=learning_curve(descriptor,prop,nonHatoms)

		# Plot learning curve
		plt.semilogx(sample_sizes,mae,'o-',color='blue')
		plt.xlabel(r'Training set size')
		plt.ylabel(r'MAE')

	elif inp3 == False:

		# Train
		y_pred,y_test=krr(descriptor,prop,nonHatoms)

		#Plot results
		plt.plot(y_test,y_pred,'.',color='blue')
		plt.plot(np.linspace(y_test.min(),y_test.max(),1000),np.linspace(y_test.min(),y_test.max(),1000),'k--')

	plt.show()

if __name__ == '__main__':
	main()