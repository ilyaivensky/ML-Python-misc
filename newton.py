#!/usr/bin/env python

import math
import numpy as np

class Newton_Raphson(object):
    
    def train(self, X, Y, w, model):
        
        def hessian(X,Etha):
            """
            X is design matrix
            Etha is a matrix of E(Y|X,w), where the number of columns is J-1 (J is the number of classes)   
            Returns -X^T*W*X, which is a matrix of second derivatives of weights likelihood 
            """
       
            h_dim = X.shape[1]*Etha.shape[1]
            H = np.zeros((h_dim, h_dim))
            for i in range(Etha.shape[1]):
                for j in range(Etha.shape[1]):
                    rf = Etha.shape[0] * i
                    rl = Etha.shape[0] * (i+1)
                    cf = Etha.shape[0] * j
                    cl = Etha.shape[0] * (j+1)  
                    if i == j:
                       W = np.diagflat([etha[:,i]*(1-etha[:,i]) for etha in Etha])
                       print 'W:', type(W).__name__, len(W)
                       print 'X:', type(X).__name__, X.shape
                       H[rf:rl,cf:cl] = -np.dot(np.dot(X.transpose(),W),X)[:,:]
                    else:
                       W = np.diagflat([etha[:,i]*etha[:,j] for etha in Etha])  
                       H[rf:rl,cf:cl] = np.dot(np.dot(X.transpose(),W),X)[:,:]
            return H

        def update(X, Y, Y_hat):
            
            h = hessian(X,Y_hat)
            hinv = np.linalg.inv(h)
            grad = np.dot(X.transpose(), Y-Y_hat)
            return np.dot(hinv, grad), np.linalg.norm(grad)
            
        normgradient = float("inf")
        i = 0
        
        while normgradient > 1e-10:
       
            Y_hat = model.predict(X,w)
            u, normgradient = update(X,Y,Y_hat)
            print 'iteration', i, 'cost', model.logistic_cost(X,Y,w), 'normgrad', normgradient
            
            w = w - u
            i += 1
            
        return w