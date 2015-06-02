#!/usr/bin/python

import data_generator
import dimension_reduction

# example usage
X = data_generator.simData(100,100,1,25)
dstar = dimension_reduction.stabilityMeasure(X,50)
U,S,V = dimension_reduction.rsvd(X,dstar)
