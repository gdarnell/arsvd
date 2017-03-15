#!/usr/bin/python

import numpy as np
from random import expovariate
from scipy import linalg

def simSphericalData(num_rows, num_cols, k, dstar):
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

def simGeneticData(num_indvs, num_snps, num_populations=10):
        """ Generate genetic data with latent population structure
	Parameters
	----------
	num_indvs : int
				total number of individuals to generate
	num_snps : int
                                number of snps (genetic variants) per individual
        num_populations : int
                                total number of latent (ancestral) populations from which
                                each individual is derived
	Returns
	-------
	int matrix
		Genetic data matrix (individuals x snps) with latent population structure
        int matrix
                Phenotype generated from latent population structure
	"""
        alpha = np.ones((num_populations,)) / num_populations
        # generate admixture proportions for each individual
        population_proportion = np.random.dirichlet(alpha, size=num_indvs)
        # allele frequencies for all snps in each population
        # snps x population_freq
        allele_freq = np.array([np.random.beta(1, 1, size=num_snps) for p in range(num_populations)]).T
        genotypes = np.zeros((num_indvs, num_snps))
        for i in range(num_indvs):
                # each individual has two latent indicators z_1 and z_2
                # for which population each SNP is derived from
                z_1 = np.random.multinomial(1, population_proportion[i], size=num_snps)
                z_2 = np.random.multinomial(1, population_proportion[i], size=num_snps)
                indv_allele_freq_1 = np.sum(allele_freq * z_1, axis=1)
                indv_allele_freq_2 = np.sum(allele_freq * z_2, axis=1)
                x_1 = np.random.binomial(1, indv_allele_freq_1)
                x_2 = np.random.binomial(1, indv_allele_freq_2)
                indv_snps = x_1 + x_2
                genotypes[i,:] = indv_snps

        phenotype = np.zeros((num_indvs,))
        for i in range(num_indvs):
                # only population "1" is affected
                likelihood = 0.5 * population_proportion[i,1] + 0.1 * (1-population_proportion[i,1])
                pheno_status = np.random.binomial(1, likelihood)
                phenotype[i] = pheno_status

        return(genotypes,phenotype)
        
