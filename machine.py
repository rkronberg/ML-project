from sklearn.kernel_ridge import KernelRidge
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE


## This part of the code reads the raw data (.xyz files) and returns the central quantities stored in arrays

def preprocess(datasize):

	# Selects all molecules with 7 or fewer non-H atoms (3963) and datasize-3963 molecules with 8 non-H atoms at random.
	# This compensates the underrepresentation of small molecules (molecules with 9 non-H atoms are excluded)

	ind = np.concatenate((np.arange(1,3964),np.random.randint(3964,21989,size=datasize-3963)))

	# Initialize the variables as empty lists
	# natoms = number of atoms in a given molecule
	# nonHatoms = number of non-H atoms in a given molecule
	# Ea = Atomization energy (Ha)
	# mu = Dipole moment (Debye
	# alpha = Isotropic polarizability (bohr^3)
	# atomlist = list of the atoms constituting a given molecule (e.g. [C,H,H,H] for methane)
	# coords = xyz coordinates of each atom in a given molecule
	# charges = Partial charges from Mulliken population analysis (e)

	natoms,nonHatoms,Ea,charges,alpha,mu,atomlist,coords=[],[],[],[],[],[],[],[]

	atomref=[-0.500273,-37.846772,-54.583861,-75.064579,-99.718730]		# Energies (Ha) of single atoms [H,C,N,O,F]
	atoms=['H','C','N','O','F']

	# Loop over all selected indices (molecules)

	for i in ind:

		xyz,elemtype,mulliken,nnonH=[],[],[],0      # Initialize list that will contain coordinates and element types of ith molecule
		i = str(i).zfill(6)     	# This pads the index with zeros so that all contain 6 digits (e.g. index 41 -> 000041)
		
		# Define the path to the .xyz file of ith molecule. Here it is assumed that the dataset is stored in a 
		# subdirectory "xyz" within the one containing machine.py

		fpath = os.path.join('xyz',"dsgdb9nsd_%s.xyz" % i)

		# Open the file and loop over the lines

		with open(fpath) as f:
			for j, line in enumerate(f):
				if j == 0:
					na = int(line)      # Number of atoms in molecule
					natoms.append(na)
				elif j == 1:
					E = float(line.split()[12])  				# Properties written on second line. Atomization energy,
					mu.append(float(line.split()[5])) 			# Dipole moment,
					alpha.append(float(line.split()[6])) 		# Polarizability
				elif 2 <= j <= na+1:
					parts = line.split()                	# Lines 2 -> na+1 contains element types, coordinates and charges
					elemtype.append(parts[0])           	# Index 0 = element type, 1 = x, 2 = y, 3 = z
					mulliken.append(parts[4])				# Partial charge on atom
					E = E - atomref[atoms.index(parts[0])]	# Subtract energy of isolated atom from total energy
					if parts[0] != 'H':
						nnonH += 1
					xyz.append(np.array([float(parts[1]),float(parts[2]),float(parts[3])]))

		Ea.append(E*27.21139)			# Convert atomization energy from Ha to eV
		atomlist.append(elemtype)
		coords.append(xyz)
		nonHatoms.append(nnonH)
		charges.append(mulliken)

	# Return all lists in the form of numpy arrays

	return np.array(natoms),np.array(Ea),np.array(mu),np.array(charges),np.array(alpha), \
		np.array(atomlist),np.array(coords),np.array(nonHatoms)


## Implement the MBTR descriptor here! Azeema?

def mbtr(mbtr_input):

	return mbtr_output


## Implement the SOAP descriptor here! Zahra?

def soap(soap_input):

	return soap_output


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
					M[i][j][k] = 0.5*Z[atoms.index(atomlist[i][j])]**2.4	# atoms.index(atomlist[i][j]) identifies the 
																			# atomtype and its index in the 'atoms' list and
																			# then gets the corresponding nuclear charge from 'Z'
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

def train(x,y,nonHatoms):

	# Split the selected data into a training set (90%) and a test set (10%), stratified using the number of non-H atoms
	# To do: (1) add learning curves, (2) optimize hyperparameters for mu and alpha, (3) figure out how to handle the partial charges

	x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,stratify=nonHatoms)
	kr = KernelRidge(kernel='laplacian',alpha=1e-11,gamma=1e-4) 		# 1e-11 and 1e-4 are optimal for CM, Ea, laplacian
	
	#kr = GridSearchCV(KernelRidge(kernel='laplacian'), cv=5,param_grid={"alpha": np.logspace(-12,0,13),"gamma": np.logspace(-6,0,7)})
	#print(kr.best_params_)		# Print optimal hyperparameters

	kr.fit(x_train,y_train)
	y_pred = kr.predict(x_test)

	print(MAE(y_test,y_pred),np.sqrt(MSE(y_test,y_pred)))		# Print mean absolute error and root mean squared error

	return y_pred,y_test


## The main routine and plotting

def main():

	datasize=10000
	natoms,Ea,mu,charges,alpha,atomlist,coords,nonHatoms = preprocess(datasize)
	Mvec = coulomb(natoms,atomlist,coords)

	Ea_pred,Ea_test=train(Mvec,Ea,nonHatoms)
	mu_pred,mu_test=train(Mvec,mu,nonHatoms)
	alpha_pred,alpha_test=train(Mvec,alpha,nonHatoms)

	# Just some plot settings
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif', size=14)

	#Plot results for Ea
	plt.figure()
	plt.plot(Ea_test,Ea_pred,'b.')
	plt.plot(np.linspace(Ea_test.min(),Ea_test.max(),1000),np.linspace(Ea_test.min(),Ea_test.max(),1000),'k--')
	plt.xlabel(r'$\Delta_\mathrm{at}E^\mathrm{DFT}$ (eV)')
	plt.ylabel(r'$\Delta_\mathrm{at}E^\mathrm{KRR}$ (eV)')

	#Plot results for mu
	plt.figure()
	plt.plot(mu_test,mu_pred,'b.')
	plt.plot(np.linspace(mu_test.min(),mu_test.max(),1000),np.linspace(mu_test.min(),mu_test.max(),1000),'k--')
	plt.xlabel(r'$\mu^\mathrm{DFT}$ (D)')
	plt.ylabel(r'$\mu^\mathrm{KRR}$ (D)')

	#Plot results for alpha
	plt.figure()
	plt.plot(alpha_test,alpha_pred,'b.')
	plt.plot(np.linspace(alpha_test.min(),alpha_test.max(),1000),np.linspace(alpha_test.min(),alpha_test.max(),1000),'k--')
	plt.xlabel(r'$\alpha^\mathrm{DFT}$ (e)')
	plt.ylabel(r'$\alpha^\mathrm{KRR}$ (e)')

	plt.show()

if __name__ == '__main__':
	main()