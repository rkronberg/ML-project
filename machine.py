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
	# u0 = internal energy at 0 K
	# atomlist = list of the atoms constituting a given molecule (e.g. [C,H,H,H] for methane)
	# coords = xyz coordinates of each atom in a given molecule

	natoms,nonHatoms,u0,atomlist,coords=[],[],[],[],[]

	# Loop over all selected indices (molecules)

	for i in ind:

		xyz,elemtype,nnonH=[],[],0      # Initialize list that will contain coordinates and element types of ith molecule
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
					u0.append(float(line.split()[12]))  # Properties written on second line. 12th index is u0
				elif 2 <= j <= na+1:
					parts = line.split()                # Lines 2 -> na+1 contains element types, coordinates and charges
					elemtype.append(parts[0])           # Index 0 = element type, 1 = x, 2 = y, 3 = z
					if parts[0] != 'H':
						nnonH += 1
					xyz.append(np.array([float(parts[1]),float(parts[2]),float(parts[3])]))

		atomlist.append(elemtype)
		coords.append(xyz)
		nonHatoms.append(nnonH)

	# Return all lists in the form of numpy arrays

	return np.array(natoms),np.array(u0),np.array(atomlist),np.array(coords),np.array(nonHatoms)


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
																			# then gets the correct nuclear charge from 'Z'
				else:
					M[i][j][k] = Z[atoms.index(atomlist[i][j])]*Z[atoms.index(atomlist[i][k])]/np.linalg.norm(coords[i][j]-coords[i][k])
		
		# Sort Coulomb matrix according to descending row norm

		indexlist = np.argsort(-np.linalg.norm(M[i],axis=1))    # Get the indices in the sorted order
		M[i] = M[i][indexlist]                                  # Rearrange the matrix
		Mvec.append(M[i][np.tril_indices(dim,k=0)])             # Convert the lower triangular matrix into a vector and append 
																# to a list of Coulomb 'vectors' 

	return Mvec

# Do the grid search (if optimal hyperparameters are not known), then training and prediction using KRR

def train(x,y,nonHatoms):

	x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,stratify=nonHatoms)
	kr = KernelRidge(kernel='laplacian',alpha=1e-12,gamma=1e-4) 				# 1e-11 and 1e-4 are optimal for CM + u0
	
	#kr = GridSearchCV(KernelRidge(kernel='laplacian'), cv=5,param_grid={"alpha": np.logspace(-12,0,13),"gamma": np.logspace(-6,0,7)})
	#print(kr.best_params_)		# Print optimal hyperparameters

	kr.fit(x_train,y_train)
	y_pred = kr.predict(x_test)

	print(MAE(y_test,y_pred),MSE(y_test,y_pred))

	return y_pred,y_test

# The main routine

def main():

	datasize=20000
	natoms,u0,atomlist,coords,nonHatoms = preprocess(datasize)
	Mvec = coulomb(natoms,atomlist,coords)

	y_pred,y_test=train(Mvec,u0,nonHatoms)

	plt.figure()
	plt.plot(y_test,y_pred,'b.')
	plt.plot(np.linspace(y_test.min(),y_test.max(),1000),np.linspace(y_test.min(),y_test.max(),1000),'k--')
	plt.show()

if __name__ == '__main__':
	main()