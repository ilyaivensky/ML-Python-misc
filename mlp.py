#!/usr/bin/env python

import numpy as np

def sigmoid(z):
    
    return 1 / (1 + np.exp(-z))

def softmax(X):
    
    """Compute the softmax of matrix X in a numerically stable way."""
    maxX = np.max(X, axis=1)
    shiftX = (X.transpose() - maxX).transpose()
    exps = np.exp(shiftX)
    return (exps.transpose() / np.sum(exps, axis=1)).transpose()

class LogisticRegression(object):
    """
    Multi-class Logistic Regression Class
    """

    def __init__(self, n_in, n_out):
        """ 
        Initialize the parameters of the logistic regression

        :param n_in: number of input units
        :param n_out: number of output units

        """
    
        self.W = np.zeros(
                (n_in, n_out),
                dtype=np.float32
            )
        
        self.b = np.zeros(
                (n_out,),
                dtype=np.float32
            )
     
    def predict(self, Z):
        
        return softmax(np.dot(Z, self.W) + self.b)
        
    
    def backprop(self, Z, Y, Y_pred, learning_rate):
        
        # gradient of loss w.r.t. to output is a matrix of size (num_examples x num_classes)
        g_err = Y_pred - Y
        prop = np.dot(g_err, self.W.transpose())
        g_W = np.dot(Z.transpose(), g_err)
        g_b = sum(g_err)
        self.W -= g_W * learning_rate
        self.b -= g_b * learning_rate
        
        return prop

    def errors(self, y, y_pred):
        
        return mean(int(y_pred_c != y_c) for y_pred, y_c in zip(y_pred, y))
    
class HiddenLayer(object):
    
    def __init__(self, rng, n_in, n_out):
        """
        Typical hidden layer of a MLP

        :param rng: a random number generator used to initialize weights
        :param n_in: dimensionality of input
        :param n_out: number of hidden units
        """
        
        self.W = np.asarray(
                rng.uniform(
                    low=-4*np.sqrt(6. / (n_in + n_out)),
                    high=4*np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=np.float32
            )
        
        self.b = np.zeros((n_out,), dtype=np.float32)

    def fwdprop(self, X):
        
        XW = np.dot(X, self.W)
        Z = sigmoid(np.dot(X, self.W) + self.b)
        return Z
    
    def backprop(self, X, Z, propagated, learning_rate):
      
        g_err = Z * (1 - Z) * propagated    
        prop = np.dot(g_err, self.W.transpose()) 
        g_W = np.dot(X.transpose(), g_err)
        g_b = sum(g_err)
        self.W -= g_W * learning_rate
        self.b -= g_b * learning_rate
        
        return prop
    

class MLP(object):
    """
    Multi-Layer Perceptron Class
    """

    def __init__(self, rng, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :param rng: a random number generator used to initialize weights
        :param n_in: number of input units
        :param n_hidden: number of hidden units
        :param n_out: number of output units

        """

        self.hiddenLayer = HiddenLayer(
            rng=rng,
            n_in=n_in,
            n_out=n_hidden
        )
        
        self.hiddenLayer2 = HiddenLayer(
            rng=rng,
            n_in=n_hidden,
            n_out=n_hidden
        )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            n_in=n_hidden,
            n_out=n_out
        ) 

    def predict(self, X):
        
        Z = self.hiddenLayer.fwdprop(X)
        Z2 = self.hiddenLayer2.fwdprop(Z)
        return self.logRegressionLayer.predict(Z2)
    
    def train(self, X, Y, learning_rate):
        
        Z = self.hiddenLayer.fwdprop(X)
        Z2 = self.hiddenLayer2.fwdprop(Z)
        Y_pred = self.logRegressionLayer.predict(Z2)
        propagated2 = self.logRegressionLayer.backprop(Z2, Y, Y_pred, learning_rate)
        propagated = self.hiddenLayer2.backprop(Z, Z2, propagated2, learning_rate)
        self.hiddenLayer.backprop(X, Z, propagated, learning_rate)
        
        return Y_pred
        
    
     


