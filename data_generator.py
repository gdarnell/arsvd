#!/usr/bin/python

import numpy as np
from random import expovariate
from scipy import linalg

def simData(num_rows, num_cols, k, dstar):
	""" Generate data with lower latent rank
	Parameters
	----------
	num_rows : int
				total rows in matrix
	num_cols : int
				total columns in matrix
	k :	int
			separation between signal and noise
	dstar : int
				latent_rank <= num_cols
				rank of latent sub-matrix to Generate
	Returns
	-------
	int matrix
		Simulated data matrix containing low rank signal
		and overlaid noise
	"""
	if(dstar > num_cols):
		dstar = num_cols
	noise = np.random.randn(num_rows,num_cols)/np.sqrt(num_rows)
	S = linalg.svd(noise)[1]
	s_1 = S[S.shape[0]-1]*k # smallest singular value
	S = np.zeros((num_cols,num_rows)) # set S to matrix of zeroes
	v_j_1 = s_1 # keep track of last singular value
	# add exponential increments to singular values
	for i in range(dstar):
		v_j = expovariate(1)
		S[i,i] = v_j + v_j_1
		v_j_1 = S[i,i]

	# generate signal
	U = linalg.svd(np.random.randn(num_rows,dstar))[0]
	V = linalg.svd(np.random.randn(dstar,num_cols))[2]
	signal = np.dot(np.dot(U,S),V.T) # U * S * V'
	
	return signal + noise
