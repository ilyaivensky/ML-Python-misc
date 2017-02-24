#!/usr/bin/env python

import numpy as np

class Fisher_LDA(object):
    
    def __init__(self,X,Y):
        
        def piMLE(Y):
            return reduce (lambda a,b: a+b, [y for y in Y]) / Y.shape[0]
            
        def muMLE(X):
            return np.mean(X, axis=0)
        
         #Split by class label
        X0 = np.array([x for x,y in zip(X,Y) if y == 0.0])
        X1 = np.array([x for x,y in zip(X,Y) if y == 1.0])
        
        #Calculate MLE
        pi = piMLE(Y)
        mu0 = muMLE(X0)
        mu1 = muMLE(X1)
        sigma = sigmaMLE(X0, X1, mu0, mu1)    
        
         
        delta_mu = mu1 - mu0
          
        self.beta = np.dot(delta_mu, 1 / sigma)
        self.gamma = np.log(pi / (1 - pi)) - np.dot(self.beta, (mu1 + mu0)) / 2.0
    
