#!/usr/bin/python

import sys
import time
import numpy as np
from scipy import linalg, stats

def rsvd(X, dstar, power_iters=2):
	""" Perform rsvd algorithm on input matrix.
		Method must be supplied dstar.
		Returns truncated svd (U,S,V).
	Parameters
	----------
	X : int matrix
    	Matrix of n x m integers, where m <= n. If n < m,
    	matrix will be transposed to enforce m <= n.
   	dstar : int
   		The latent (underlying) matrix rank that will be
   		used to truncate the larger dimension (m).
   	power_iters : int
   		default: 2
   		Number of power iterations used (random matrix multiplications)
    Returns
	-------
	int matrix
    	Matrix of left singular vectors.
    int matrix
    	Matrix of singular values.
    int matrix
    	Matrix of right singular vectors.
    """
	if(X.shape[0] < X.shape[1]):
		X = X.T  # transpose X
	if(power_iters < 1):
		power_iters = 1
	# follows manuscript notation as closely as possible
	P = np.random.randn(X.shape[1],dstar)
	for i in range(power_iters):
		P = np.dot(X.T,P)
		P = np.dot(X,P)
	Q,R = np.linalg.qr(P)
	B = np.dot(Q.T,X)
	U,S,V = linalg.svd(B)
	U = np.dot(Q,U)
	return U,S,V

def stabilityMeasure(X, d_max, B=5, power_iters=2):
	""" Calculate stability of 
	Parameters
	----------
	X : int matrix
		input matrix to determine rank of
	d_max : int
		upper bound rank to estimate
	B : int
		default: 5
		number of projections to correlate
	power_iters : int
		default: 2
   		Number of power iterations used (random matrix multiplications)
	Returns
	-------
	int
		Latent (lower-dimensional) matrix rank
	"""
	singular_basis = np.zeros((B,X.shape[0],d_max))
	# calculate singular basis under multiple projections
	for i in range(B):
		U = rsvd(X,d_max)[0]
		singular_basis[i,:,:] = U[:,0:d_max]

	# calculate score for each singular vector
	stability_vec = np.zeros((d_max))
	for k in range(d_max):
		stability = 0
		for i in range(0,B-1):
			for j in range(i+1,B):
				corr = stats.spearmanr(singular_basis[i,:,k],singular_basis[j,:,k])[0]
				stability = stability + abs(corr)
		N = B*(B-1)/2
		stability = stability/N
		stability_vec[k] = stability

	# wilcoxon rank-sum test p-values
	p_vals = np.zeros(d_max-2)
	for k in range(2,d_max):
		p_vals[k-2] = stats.ranksums(stability_vec[0:k-1],stability_vec[k-1:d_max])[1]

	dstar = np.argmin(p_vals)
	
	return dstar
