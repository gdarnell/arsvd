#!/usr/bin/python

import data_generator
import dimension_reduction

# example usage for spherical data
X = data_generator.simSphericalData(100,100,1,25)
dstar = dimension_reduction.stabilityMeasure(X,50)
U,S,V = dimension_reduction.rsvd(X,dstar)

# example usage for genetic data
genotypes,phenotype = data_generator.simGeneticData(100,500)
