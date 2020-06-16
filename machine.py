import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from scipy.special import comb
from itertools import combinations, permutations

## This part of the code reads the raw data (.xyz files) and returns the central quantities stored in arrays
def preprocess(datasize,atoms):

	# Selects all molecules with 7 or fewer non-H atoms (3963) and (datasize - 3963) molecules with 8 non-H atoms at random.
	# This compensates the underrepresentation of small molecules (molecules with 9 non-H atoms are excluded)
	ind = np.concatenate((np.arange(1,3964),np.random.randint(3964,21989,size=datasize-3963)))

	# Initialize the variables as empty lists
	# natoms = number of atoms in a given molecule
	# nonHatoms = number of non-H atoms in a given molecule 21989
	# Ea = Atomization energy (Ha)
	# dipmom = Dipole moment (D)
	# polar = Isotropic polarizability (bohr^3)
	# atomlist = list of the atoms constituting a given molecule (e.g. ['C','H','H','H'] for methane)
	# coords = xyz coordinates of each atom in a given molecule

	natoms,nonHatoms,Ea,polar,dipmom,gap,atomlist,coords=[],[],[],[],[],[],[],[]
	atomref=[-0.500273,-37.846772,-54.583861,-75.064579,-99.718730]     # Energies (Ha) of single atoms [H,C,N,O,F]

	# Loop over all selected indices (molecules)
	for i in ind:
		# Initialize list that will contain coordinates and element types of ith molecule
		xyz,elemtype,mulliken,nnonH=[],[],[],0
		# This pads the index with zeros so that all contain 6 digits (e.g. index 41 -> 000041)
		i = str(i).zfill(6)
		
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
					dipmom.append(float(line.split()[5])*0.20819)   # Dipole moment,
					polar.append(float(line.split()[6])*0.14818)    # Polarizability
					gap.append(float(line.split()[9])*27.21139)     # HOMO-LUMO gap
				elif 2 <= j <= na+1:
					parts = line.split()                    # Lines 2 -> na+1 contains element types, coordinates and charges
					elemtype.append(parts[0])               # Index 0 = element type, 1 = x, 2 = y, 3 = z
					E = E - atomref[atoms.index(parts[0])]  # Subtract energy of isolated atom from total energy
					if parts[0] != 'H':
						nnonH += 1
					xyz.append(np.array([float(parts[1]),float(parts[2]),float(parts[3])]))

		Ea.append(-E*27.21139)
		atomlist.append(elemtype)
		coords.append(xyz)
		nonHatoms.append(nnonH)

	# Return all lists in the form of numpy arrays
	return np.array(natoms),np.array(Ea),np.array(dipmom),np.array(polar),np.array(gap), \
		np.array(atomlist),np.array(coords),np.array(nonHatoms)

def gauss(x,weight,sigma,mu):

	return weight/(sigma*np.sqrt(2*np.pi))*np.exp(-((x-mu)**2)/(2*sigma**2))

def mbtr(atomlist,coords,atoms,Z):

	# Decay factor (d) and sigmas are roughly optimal
	d=0.5
	w1=1
	sigma1,sigma2,sigma3=0.1,0.01,0.05
	x1=np.linspace(0,10,201)
	x2=np.linspace(0,1.25,201)
	x3=np.linspace(-1,1,201)
	mbtr_output=[]
	
	atoms = list(set([''.join(p) for p in combinations('CHONF',1)]))
	pairs = list(set([''.join(p) for p in combinations('CCHHOONNFF',2)]))
	triples = list(set([''.join(p) for p in permutations('CCCHHHOOONNNFFF',3)]))

	for i in range(len(atomlist)):
		bag1=dict((k,np.zeros(len(x1))) for k in atoms)
		bag2=dict((k,np.zeros(len(x2))) for k in pairs) 
		bag3=dict((k,np.zeros(len(x3))) for k in triples)
		MBTRvec=np.array([]) 	
		for j in range(len(atomlist[i])):
			g1=Z[atoms.index(atomlist[i][j])]
			bag1[atomlist[i][j]]+=gauss(x1,w1,sigma1,g1)
			for k in range(len(atomlist[i])):
				if k > j:
					Rjk=np.linalg.norm(coords[i][j]-coords[i][k])
					w2=np.exp(-d*Rjk)
					g2=1/Rjk
					try:
						bag2[atomlist[i][j]+atomlist[i][k]]+=gauss(x2,w2,sigma2,g2)	
					except KeyError:
						bag2[atomlist[i][k]+atomlist[i][j]]+=gauss(x2,w2,sigma2,g2)
					for l in range(len(atomlist[i])):
						if l > k:
							Rjl=np.linalg.norm(coords[i][j]-coords[i][l])
							Rkl=np.linalg.norm(coords[i][k]-coords[i][l])
							w3=np.exp(-d*(Rjk+Rjl+Rkl))
							g3=np.dot(coords[i][j]-coords[i][l],coords[i][k]-coords[i][l])/(Rjl*Rkl)
							try:
								bag3[atomlist[i][j]+atomlist[i][l]+atomlist[i][k]]+=gauss(x3,w3,sigma3,g3)	
							except KeyError:
								bag3[atomlist[i][k]+atomlist[i][l]+atomlist[i][j]]+=gauss(x3,w3,sigma3,g3)

		for atom in bag1:
			MBTRvec = np.concatenate((MBTRvec,bag1[atom]))
		for pair in bag2:
			MBTRvec = np.concatenate((MBTRvec,bag2[pair]))
		for triple in bag3:
			MBTRvec = np.concatenate((MBTRvec,bag3[triple]))

		mbtr_output.append(MBTRvec)

	return mbtr_output

## The BoB descriptor
def bob(atomlist,coords,atoms,Z):

	bob_output = []
	# 18 H atoms in octane -> comb(18,2) H-H pairs (max. size of a bond vector in a bag of bonds)
	dim = int(comb(18,2))
	perms = list(set([''.join(p) for p in combinations('CCHHOONNFF',2)]))

	for i in range(len(atomlist)):
		bag=dict((k,dim*[0]) for k in perms)
		BoBvec = np.array([])
		for j in range(len(atomlist[i])):
			for k in range(len(atomlist[i])):
				if j > k:
					try:
						bag[atomlist[i][j]+atomlist[i][k]].insert(0,Z[atoms.index(atomlist[i][j])]*Z[atoms.index(atomlist[i][k])]/np.linalg.norm(coords[i][j]-coords[i][k]))
						del bag[atomlist[i][j]+atomlist[i][k]][-1]
					except KeyError:
						bag[atomlist[i][k]+atomlist[i][j]].insert(0,Z[atoms.index(atomlist[i][j])]*Z[atoms.index(atomlist[i][k])]/np.linalg.norm(coords[i][j]-coords[i][k]))	
						# Avoid KeyError raised by "wrong" order of atoms in a bond (e.g. 'CH' -> 'HC')
						del bag[atomlist[i][k]+atomlist[i][j]][-1]	
						
		for pair in bag:
			BoBvec = np.concatenate((BoBvec,np.array(sorted(bag[pair],reverse=True))))

		bob_output.append(BoBvec)

	return bob_output

## The following function takes the number of atoms in each molecule, the atom types and corresponding coordinates 
## and returns an array of corresponding Coulomb matrices
def coulomb(natoms,atomlist,coords,atoms,Z):

	dim = natoms.max()                          # Specify the dimensions of the Coulomb matrices based on the largest molecule
	CM = np.zeros((len(natoms),dim,dim))        # Initialize an array of all Coulomb matrices
	CMvec = []

	for i in range(len(natoms)):                # Loop over all molecules
		for j in range(len(atomlist[i])):   # Loop over all atom pairs (j,k) in molecule i
			for k in range(len(atomlist[i])):
				if j == k:
					CM[i][j][k] = 0.5*Z[atoms.index(atomlist[i][j])]**2.4
				else:
					CM[i][j][k] = Z[atoms.index(atomlist[i][j])]*Z[atoms.index(atomlist[i][k])]/np.linalg.norm(coords[i][j]-coords[i][k])
		
		# Sort Coulomb matrix according to descending row norm
		indexlist = np.argsort(-np.linalg.norm(CM[i],axis=1))     # Get the indices in the sorted order
		CM[i] = CM[i][indexlist]                                  # Rearrange the matrix
		CMvec.append(CM[i][np.tril_indices(dim,k=0)])             # Convert the lower triangular matrix into a vector and append 
		                                                          # to a list of Coulomb 'vectors' 

	return CMvec

## Do the grid search (if optimal hyperparameters are not known), then training and prediction using KRR
## If doing grid search for optimal parameters use small training set size, like 1k (takes forever otherwise)
def krr(x,y,nonHatoms):

	inp4 = input('Do grid search for optimal hyperparameters? [True/False]\n')

	if inp4 == True:
		inp5 = raw_input('Provide kernel. [laplacian/rbf]\n').split()
		x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.9,stratify=nonHatoms)
		kr = GridSearchCV(KernelRidge(kernel=inp5[0]),cv=5,param_grid={"alpha": np.logspace(-11,-1,11),"gamma": np.logspace(-9,-3,7)})
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
	r2 = R2(y_test,y_pred)

	# Print mean absolute error and root mean squared error

	print('Mean absolute error: ' + repr(mae) + ', Root mean squared error: ' + repr(rmse) + ', R2-score: ' + repr(r2))

	return y_pred,y_test

def learning_curve(x,y,nonHatoms):

	# Do training with different sample sizes and see how the MAE behaves (learning curve)
	inp5 = raw_input('Provide kernel and hyperparameters. [kernel alpha gamma]\n').split()

	mae,rmse,r2=[],[],[]
	sample_sizes = [50,200,1000,3000,9000]
	kr = KernelRidge(kernel=inp5[0],alpha=float(inp5[1]),gamma=float(inp5[2]))

	for i in sample_sizes:
		x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=1-float(i)/len(y),stratify=nonHatoms)
		kr.fit(x_train,y_train)
		y_pred = kr.predict(x_test)
		mae.append(MAE(y_test,y_pred))
		rmse.append(np.sqrt(MSE(y_test,y_pred)))
		r2.append(R2(y_test,y_pred))

		print('Mean absolute error: ' + repr(mae[-1]) + ', Root mean squared error: ' + repr(rmse[-1]) + ', R2-score: ' + repr(r2[-1]))

	return y_pred,y_test,mae,rmse,sample_sizes

## The main routine and plotting
def main():

	# Just some plot settings
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif', size=14)
	plt.rc('xtick', direction='in')

	# Preprocess data
	datasize=10000
	atoms = ['H','C','N','O','F']
	Z = [1,6,7,8,9]
	natoms,Ea,dipmom,polar,gap,atomlist,coords,nonHatoms = preprocess(datasize,atoms)

	inp1 = raw_input('Which descriptor? [CM/BoB/MBTR]\n')

	if inp1 == 'CM':
		descriptor = coulomb(natoms,atomlist,coords,atoms,Z)
	
	elif inp1 == 'BoB':
		descriptor = bob(atomlist,coords,atoms,Z)

	elif inp1 == 'MBTR':
		descriptor = mbtr(atomlist,coords,atoms,Z)

	inp2 = raw_input('Which property? [Ea/gap/polar/dipmom]\n')

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
		
	elif inp2 == 'polar':
		prop = polar
		plt.title(r'Isotropic polarizability (\r{A}$^3$)')
		plt.xlabel(r'$\alpha^\mathrm{DFT}$ (\r{A}$^3$)')
		plt.ylabel(r'$\alpha^\mathrm{KRR}$ (\r{A}$^3$)')

	elif inp2 == 'dipmom':
		prop = dipmom
		plt.title(r'Dipole moment (e\r{A})')
		plt.xlabel(r'$\mu^\mathrm{DFT}$ (e\r{A})')
		plt.ylabel(r'$\mu^\mathrm{KRR}$ (e\r{A})')

	inp3 = input('Plot learning curve? [True/False]\n')

	if inp3 == True:
		# Train
		y_pred,y_test,mae,rmse,sample_sizes=learning_curve(descriptor,prop,nonHatoms)
		np.savetxt('dipmom_BoB.dat',np.c_[y_test,y_pred])
		np.savetxt('dipmom_BoB_lc.dat',np.c_[sample_sizes,mae])
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

	## Optimal hyperparameters for CM + Laplacian kernel
	# Ea: 			alpha 1e-11, gamma 1e-4
	# polarizability: 	alpha 1e-3, gamma 1e-4
	# HOMO-LUMO gap: 	alpha 1e-2, gamma 1e-4
	# Dipole moment: 	alpha 1e-1, gamma 1e-3

	## Optimal hyperparameters for BoB + Laplacian kernel
	# Ea: 			alpha 1e-11, gamma 1e-5
	# polarizability: 	alpha 1e-3, gamma 1e-4
	# HOMO-LUMO gap: 	alpha 1e-3, gamma 1e-4
	# Dipole moment: 	alpha 1e-1, gamma 1e-3

	## Optimal hyperparameters for MBTR + Gaussian kernel
	# Ea:			alpha 1e-7, gamma 1e-8
	# polarizability:	alpha 1e-6, gamma 1e-7
	# HOMO-LUMO gap:	alpha 1e-3, gamma 1e-6
	# Dipole moment:	alpha 1e-2, gamma 1e-5


	## Results for CM + Laplacian kernel
	# Ea: 			MAE 0.38,	RMSE 0.55,	R2 0.9977
	# polarizability: 	MAE 0.12,	RMSE 0.18,	R2 0.9828
	# HOMO-LUMO gap: 	MAE 0.56,	RMSE 0.70,	R2 0.7203
	# Dipole moment: 	MAE 0.14,	RMSE 0.19,	R2 0.5901

	## Results for BoB + Laplacian kernel
	# Ea: 			MAE 0.08,	RMSE 0.13,	R2 0.9998
	# polarizability: 	MAE 0.06,	RMSE 0.09,	R2 0.9952
	# HOMO-LUMO gap: 	MAE 0.23,	RMSE 0.31,	R2 0.9465
	# Dipole moment: 	MAE 0.11,	RMSE 0.16,	R2 0.7327

	## Results for MBTR + Gaussian kernel
	# Ea: 			MAE 0.04,	RMSE 0.06,	R2 0.9999
	# polarizability: 	MAE 0.02,	RMSE 0.04,	R2 0.9993
	# HOMO-LUMO gap: 	MAE 0.17,	RMSE 0.23,	R2 0.9686
	# Dipole moment: 	MAE 0.08,	RMSE 0.11,	R2 0.8508
