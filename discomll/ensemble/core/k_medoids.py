"""
Special purpose k - medoids algorithm 
"""

import numpy as np

def fit(sim_mat, D_len, cidx):
	"""
	Algorithm maximizes energy between clusters, which is distinction in this algorithm. Distance matrix contains mostly 0, which are overlooked due to search of maximal distances. Algorithm does not try to retain k clusters. 

	D: numpy array - Symmetric distance matrix
	k: int - number of clusters
	"""
	
	min_energy = np.inf
	for j in range(3):
		#select indices in each sample that maximizes its dimension
		inds = [np.argmin([sim_mat[idy].get(idx, 0) for idx in cidx]) for idy in range(D_len) if idy in sim_mat]
	
		cidx = []
		energy = 0 #current enengy
		for i in np.unique(inds):
			indsi = np.where(inds == i)[0] #find indices for every cluster				
			
			minind, min_value = 0, 0
			for index, idy in enumerate(indsi):
				if idy in sim_mat:
					#value = sum([sim_mat[idy].get(idx,0) for idx in indsi])
					value = 0
					for idx in indsi:
						value += sim_mat[idy].get(idx,0)
					if value < min_value:
						minind, min_value = index, value
			energy += min_value
			cidx.append(indsi[minind]) #new centers

		if energy < min_energy:
			min_energy, inds_min, cidx_min = energy, inds, cidx	

	return inds_min, cidx_min #cluster for every instance, medoids indices















