from sklearn.kernel_ridge import KernelRidge
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve

## This part of the code reads the raw data (.xyz files) and returns the central quantities stored in arrays

def preprocess(dataset):

	# Select randomly a number of indices defined by 'dataset'
	# This needs to be improved since the data is not homogeneous

	ind=np.random.randint(1,133886,size=dataset)

	# Initialize the variables as empty lists
	# natoms = number of atoms in a given molecule
	# u0 = internal energy at 0 K
	# atomlist = list of the atoms constituting a given molecule (e.g. [C,H,H,H] for methane)
	# coords = xyz coordinates of each atom in a given molecule

	natoms,u0,atomlist,coords=[],[],[],[]

	# Loop over all selected indices (molecules)

	for i in ind:

		xyz,elemtype=[],[]		# Initialize list that will contain coordinates and element types of ith molecule
		i = str(i).zfill(6)		# This pads the index with zeros so that all contain 6 digits (e.g. index 41 -> 000041)
		
		# Define the path to the .xyz file of ith molecule. Here it is assumed that the	dataset is stored in a 
		# subdirectory "xyz" within the one containing machine.py

		fpath = os.path.join('xyz',"dsgdb9nsd_%s.xyz" % i)

		# Open the file and loop over the lines

		with open(fpath) as f:
			for j, line in enumerate(f):
				if j == 0:
					na = int(line)		# Number of atoms in molecule
					natoms.append(na)
				elif j == 1:
				 	u0.append(float(line.split()[12]))	# Properties written on second line. 12th index is u0
				elif 2 <= j <= na+1:
					parts = line.split()				# Lines 2 -> na+1 contains element types, coordinates and charges
					elemtype.append(parts[0])			# Index 0 = element type, 1 = x, 2 = y, 3 = z
					xyz.append(np.array([float(parts[1]),float(parts[2]),float(parts[3])]))

		atomlist.append(elemtype)
		coords.append(xyz)

	# Return all lists in the form of numpy arrays

	return np.array(natoms),np.array(u0),np.array(atomlist),np.array(coords)


## Implement the MBTR descriptor here!

def mbtr(mbtr_input):

	return mbtr_output


## Implement the SOAP descriptor here!

def soap(soap_input):

	return soap_output


## The following function takes the number of atoms in each molecule, the atom types and corresponding coordinates 
## and returns an array of corresponding Coulomb matrices

def coulomb(natoms,atomlist,coords):

	dim = natoms.max()						# Specify the dimensions of the Coulomb matrices based on the largest molecule
	atoms = ['C','H','O','N','F']			# List of all possible atom types
	Z = [6,1,8,7,9]							# The corresponding nuclear charges
	M = np.zeros((len(natoms),dim,dim))		# Initialize an array of all Coulomb matrices
	Mvec = []

	for i in range(len(natoms)):				# Loop over all molecules
		for j in range(len(atomlist[i])):		# Loop over all atom pairs (j,k) in molecule i
			for k in range(len(atomlist[i])):
				if j == k:
					M[i][j][k] = 0.5*Z[atoms.index(atomlist[i][j])]**2.4
				else:
					M[i][j][k] = Z[atoms.index(atomlist[i][j])]*Z[atoms.index(atomlist[i][k])]/np.linalg.norm(coords[i][j]-coords[i][k])
		
		# Sort Coulomb matrix according to descending row norm

		indexlist = np.argsort(-np.linalg.norm(M[i],axis=1))	# Get the indices in the sorted order
		M[i] = M[i][indexlist]									# Rearrange the matrix
		Mvec.append(M[i][np.tril_indices(dim,k=0)])				# Convert the lower triangular matrix into a vector and append 
																# to a list of Coulomb 'vectors' 

	return Mvec

# Do the grid search of optimal hyperparameters, training and prediction using KRR

def train(Mvec,u0,trainsize):

	kr = GridSearchCV(KernelRidge(kernel='laplacian'), cv=5,param_grid={"alpha": np.logspace(-9,0,10),"gamma": np.logspace(-7,2,10)})
	kr.fit(Mvec[:trainsize],u0[:trainsize])
	u0_krr=kr.predict(Mvec[trainsize:-1])

	return u0_krr

# The main routine

def main():

	dataset=10000
	natoms,u0,atomlist,coords = preprocess(dataset)
	Mvec = coulomb(natoms,atomlist,coords)

	trainsize=int(0.1*dataset)
	u0_krr=train(Mvec,u0,trainsize)

	plt.figure()
	plt.plot(u0[trainsize:-1],u0_krr,'b.')
	plt.plot(np.linspace(u0.min(),u0.max(),1000),np.linspace(u0.min(),u0.max(),1000),'k--')
	plt.show()

if __name__ == '__main__':
	main()