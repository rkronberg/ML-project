from sklearn.kernel_ridge import KernelRidge
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve

def preprocess(dataset):

	ind=np.random.randint(1,133886,size=dataset)
	natoms,u0,atomlist,coords=[],[],[],[]

	for i in ind:
		xyz,elemtype=[],[]
		i = str(i).zfill(6)
		fpath = os.path.join('xyz',"dsgdb9nsd_%s.xyz" % i)

		with open(fpath) as f:
			for j, line in enumerate(f):
				if j == 0:
					na = int(line)
					natoms.append(na)
				elif j == 1:
				 	u0.append(float(line.split()[12]))
				elif 2 <= j <= na+1:
					parts = line.split()
					elemtype.append(parts[0])
					xyz.append(np.array([float(parts[1]),float(parts[2]),float(parts[3])]))

		atomlist.append(elemtype)
		coords.append(xyz)

	return np.array(natoms),np.array(u0),np.array(atomlist),np.array(coords)

def coulomb(natoms,atomlist,coords):

	dim = natoms.max()
	atoms = ['C','H','O','N','F']
	Z = [6,1,8,7,9]
	M = np.zeros((len(natoms),dim,dim))
	Mvec = []

	for i in range(len(natoms)):
		for j in range(len(atomlist[i])):
			for k in range(len(atomlist[i])):
				if j == k:
					M[i][j][k] = 0.5*Z[atoms.index(atomlist[i][j])]**2.4
				else:
					M[i][j][k] = Z[atoms.index(atomlist[i][j])]*Z[atoms.index(atomlist[i][k])]/np.linalg.norm(coords[i][j]-coords[i][k])
		indexlist = np.argsort(-np.linalg.norm(M[i],axis=1))
		M[i] = M[i][indexlist]
		Mvec.append(M[i][np.tril_indices(dim,k=0)])

	return Mvec

def train(Mvec,u0,trainsize):

	kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,param_grid={"alpha": np.logspace(-9,0,10),"gamma": np.logspace(-7,2,10)})
	kr.fit(Mvec[:trainsize],u0[:trainsize])
	u0_krr=kr.predict(Mvec[trainsize:-1])

	return u0_krr

def main():

	dataset=10000
	natoms,u0,atomlist,coords = preprocess(dataset)
	Mvec = coulomb(natoms,atomlist,coords)

	trainsize=int(0.2*dataset)
	u0_krr=train(Mvec,u0,trainsize)

	plt.figure()
	plt.plot(u0[trainsize:-1],u0_krr,'b.')
	plt.plot(np.linspace(u0.min(),u0.max(),1000),np.linspace(u0.min(),u0.max(),1000),'k--')
	plt.show()

if __name__ == '__main__':
	main()