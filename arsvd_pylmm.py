#!/usr/bin/python

from __future__ import print_function
import sys

import numpy as np
from pylmm import lmm
from pylmm import input
from scipy import linalg, stats
import dimension_reduction
import data_generator

def save_result(p_values,d_star,num_indvs,num_snps,false_positives,power,output_file):
    for p in p_values:
        str_out = str(num_indvs) +"\t"+ str(num_snps) +"\t"+ str(false_positives) +"\t"+ str(power) +"\t"+ str(d_star) +"\t"+ str(p)
        print(str_out,file=output_file)

def pvalue_corr(ps1,ps2):
    sum = 0
    for i in range(len(ps1)):
        if(not(np.isnan(ps1[i])) and int(ps1[i]*10**6) == int(ps2[i]*10**6)):
            sum = sum + 1
        if(np.isnan(ps1[i]) and np.isnan(ps2[i])):
            sum = sum + 1
    print(sum)

    total = 0.0
    sig = 0.0
    recall = 0.0
    false_positives = 0.0
    for i in range(len(ps2)):
        if(not np.isnan(ps2[i])):
            total += 1
            if(ps2[i] < 0.01):
                sig += 1
                if(ps1[i] < 0.01):
                    recall += 1
            if(ps1[i] < 0.01):
                false_positives += 1
    #false_positives -= recall

    print("total:",total)
    print("sig:",sig)
    print("recall:",recall)
    print("fp:",false_positives)
    print('spearman corr: ',stats.spearmanr(ps1,ps2))
    print('pearson corr: ',stats.pearsonr(ps1,ps2))
    fpprecent = 0.0
    if(total != 0):
        fpprecent = false_positives/total
    power = 0.0
    if(sig != 0):
        power = recall/sig
    return (fpprecent,power)

num_individuals = 100
num_snps = 500

genotypes,phenotype = data_generator.simGeneticData(num_individuals,num_snps)

np.savetxt("geno_n_"+str(num_individuals)+"_p_"+str(num_snps)+".txt",genotypes,fmt='%d')
np.savetxt("pheno_n_"+str(num_individuals)+"_p_"+str(num_snps)+".txt",phenotype,fmt='%d')

print('loading genos...')
snp_filename = "geno_n_" + str(num_individuals) + "_p_" + str(num_snps) + ".txt"
snps = np.genfromtxt(snp_filename)
print('loading phenos...')
pheno_filename = "pheno_n_" + str(num_individuals) + "_p_" + str(num_snps) + ".txt"
Y = np.genfromtxt(pheno_filename)

print('phenos shape: ' + str(Y.shape))
print('genos shape: ' + str(snps.shape))

X = snps
normGenos = False
if(normGenos):
    print('begin snp norm filter')
    for i in range(X.shape[1]):
        tmp = X[:,i]
        tmpmean = tmp[True - np.isnan(tmp)].mean()
        tmpvar = np.sqrt(tmp[True - np.isnan(tmp)].var())
        if(tmpvar == 0):
            X[:,i] = tmpmean
        else:
            for j in range(X.shape[0]):
                if(np.isnan(X[j,i])):
                    X[j,i] = tmpmean
                else:
                    X[j,i] = (X[j,i] - tmpmean) / tmpvar
                    print('end snp norm filter')

### EIG
print('\neigen decomposition')
K = np.zeros((snps.shape[0],snps.shape[0])) * np.nan
kin = np.dot(X,np.transpose(X))
Kva,Kve = np.linalg.eigh(kin)
print('execute gwas...')
ts2,ps2 = lmm.GWAS(Y,snps,K,Kva=Kva,Kve=Kve,REML=False,refit=False)
(false_positives,power) = pvalue_corr(ps2,ps2)
output_filename = "eig_pylmm_n_" + str(num_individuals) + "_p_" + str(num_snps) + ".txt"
output_file = open(output_filename,'w')
save_result(ps2,num_snps,num_individuals,num_snps,false_positives,power,output_file)

### ARSVD
print('\nARSVD')
dstar = dimension_reduction.stabilityMeasure(X,80)
U,S,V = dimension_reduction.rsvd(X,dstar)
S = np.multiply(S,S) # convert singular values to eigenvalues
if(snps.shape[0] < snps.shape[1] and dstar < U.shape[0]):
    extradim = U.shape[0] - dstar
    S = np.concatenate((S,np.zeros((extradim,))),axis=0) # add dimensionality
elif(snps.shape[0] >= snps.shape[1] and dstar < U.shape[0]):
    extradim = U.shape[0] - dstar
    U = np.hstack([U,np.zeros((X.shape[0],extradim))]) # add dimensionality
    S = np.concatenate((S,np.zeros((extradim,))),axis=0) # add dimensionality
print('execute gwas...')
ts,ps1 = lmm.GWAS(Y,snps,K,Kva=S,Kve=U,REML=False,refit=False)
(false_positives,power) = pvalue_corr(ps1,ps2)
output_filename = "arsvd_pylmm_n_" + str(num_individuals) + "_p_" + str(num_snps) + ".txt"
output_file = open(output_filename,'w')
save_result(ps1,dstar,num_individuals,num_snps,false_positives,power,output_file)
