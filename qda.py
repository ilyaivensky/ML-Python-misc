#!/usr/bin/env python

import numpy as np

class QDA():
    """
    Quadratic discriminant analysis
    """
    
    def __init__(self, X, Y):
         
        def piMLE(Y):
            return reduce (lambda a,b: a+b, [np.asscalar(y) for y in Y]) / Y.shape[0]
            
        def muMLE(X):
            return np.mean(X, axis=0)
        
        def sigmaMLE(X, mu):
            return reduce(lambda a,b: a+b, [np.dot((x-mu).transpose(),(x-mu)) for x in X]) / X.shape[0]
        
         #Split by class label
        X0 = np.matrix(np.array([x for x,y in zip(X,Y) if y == 0.0]))
        X1 = np.matrix(np.array([x for x,y in zip(X,Y) if y == 1.0]))
    
        pi = piMLE(Y)
        mu0 = muMLE(X0)
        mu1 = muMLE(X1)
        sigma0 = sigmaMLE(X0, mu0) 
        sigma1 = sigmaMLE(X1, mu1)   
         
        # Some cache 
        sigma_inv_0 = np.linalg.inv(sigma0)
        sigma_inv_1 = np.linalg.inv(sigma1)
        mu_sigma0 = np.dot(mu0, sigma_inv_0)
        mu_sigma1 = np.dot(mu1, sigma_inv_1)
    
        self.alpha = -(sigma_inv_1 - sigma_inv_0) / 2 
        self.beta = mu_sigma1 - mu_sigma0
        self.gamma = np.log(pi / (1 - pi)) - ((np.dot(mu_sigma1, mu1.transpose()) - np.dot(mu_sigma0, mu0.transpose())) / 2.0)
        
 