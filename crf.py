#!/usr/bin/env python

import numpy as np

from scipy import optimize

from multiprocessing.pool import ThreadPool

def log_add(a,b):
    
    if a == -float("inf") or b == -float("inf"):
        return -float("inf") 
    
    if a > b:
        return a + np.log(1+np.exp(b-a))
    return b + np.log(1+np.exp(a-b))

def sum_log(X):
    
    total = X[0]
    for x in X[1:]:
        total = log_add(total, x) 
    
    return total

class LinearChainCRF(object):
    """
    Linear-chain CRF, optimized with LBFGS algorithm (scipy implementation).
    """
    
    def __init__(self, S, f, num_t_sens, num_t_insens):
        """ Initialize the parameters of CRF

        :param S : num of states
        :param f : feature function
         
        Code optimization parameters - help to disable evaluation of feature functions,
        which return 0 for current t 
        (see log_psi() and gradient_impl())
        
        :param num_t_sens : num of t-sensitive feature functions
        :param num_t_insens : num of t-insensitive feature functions
        
        """
        # Number of states 
        self.S = S 
        self.f = f
        self.t_sens = num_t_sens
        self.t_insens = num_t_insens
        
        self.name = 'CRF'
   
    def numStates(self):
        return self.S
    
    def classify(self, X, theta):
    
        (gamma, xi), logZ = self.expect(X, theta)
        Z = np.argmax(gamma, axis=1)
        return Z, gamma, logliks
    
    def predict(self, X, theta):
    
        (gamma, xi), logZ = self.expect(X, theta)
        return gamma
    
    def log_likelihood(self, X, Y, theta):
        """
        Returns log-likelihood of theta
        """
        
        alphas, logZ = self.alpha_chain(X, theta)
        total = 0
        
        s_prev = self.S
        
        for t,s in enumerate(Y):
            total += self.log_psi(theta, t, s_prev, s, X[t]) 
            s_prev = s
                    
        total -= logZ
        return total / X.shape[0] 
    
    def logistic_cost(self, X,Y,theta):
        
        T = X.shape[0]
        (gamma, xi), logZ = self.expect(X, theta)
         
        cost = 0
        for t,y_hat in enumerate(gamma):
            correct = 0
            total = 0
            for j,g in enumerate(y_hat):
                exp_g = np.exp(g)
                if j == Y[t]: 
                    correct = g
                total += exp_g 
            cost += correct - np.log(total)
      
        cost = -cost / T
        grad = self.gradient(gamma, xi, X, Y, theta)
#         normgrad = np.linalg.norm(grad)
#         print 'cost', cost, 'normgrad', normgrad
        return cost, grad
        
        
#         grad = self.gradient(gamma, xi)
#         print 'cost', cost, 'normgrad', np.linalg.norm(grad) 
#         return cost, grad
        
    def decode(self, X, theta):
        """
        Viterbi decoding - the same as gamma(t) for each t, but much faster
        """
        
        T = X.shape[0]
        V = np.zeros((T,self.S), dtype=[('v',np.float32),('b',np.int32)])
   
        V[0]['v'] = np.array([self.log_psi(theta, 0,self.S,s, X[0]) for s in range(self.S)])
    
        for t in range(1, T):
            for j in range(self.S):
                V[t,j]['v'], V[t,j]['b'] = max((V[t-1,i]['v']+self.log_psi(theta, t,i,j, X[t]), i) for i in range(self.S))
         
        # Restore the path by tracing back
        path = np.zeros(T, dtype=np.int32)               
        path[T-1] = np.argmax([V[T-1,j]['v'] for j in range(self.S)])
        
        for t in range(T-1,0,-1):
            path[t-1] = V[t,path[t]]['b']
        
        # None is for compatibility with classify(X)    
        return path, None, V[-1,path[-1]]['v']
    
    def psi(self, theta, t, i, j, x):
            
        return np.exp(self.log_psi(theta, t, i, j, x))
    
    def log_psi(self, theta, t, i, j, x):
           
        begin = t*self.t_sens
        end = begin+self.t_sens
        
        return sum( [ tk*fk(t,i,j, x.A1) for tk,fk in zip(theta[begin:end], self.f[begin:end]) ] )+sum( [ tk*fk(t,i,j, x.A1) for tk,fk in zip(theta[-self.t_insens:], self.f[-self.t_insens:]) ] )
    
    def alpha_chain(self, X, theta):
        """
        Calculates joint probability of sequence x_{0}, ..., x_{t} and being at state k at time t
        (i.e. p(x_{0}, ...,x_{t},z_{t}=k) 
        """  
        T = X.shape[0]
        
        alphas = np.zeros((T, self.S))
        
        for t in range(T):
            local_alpha = np.zeros(self.S)
            for j in range(self.S):
                if t == 0:
                    local_alpha[j] = self.log_psi(theta, t,self.S,j,X[t])
                else:
                    local_alpha[j] = sum_log([alphas[t-1,i] + self.log_psi(theta, t,i,j,X[t]) for i in range(self.S)])
        
            alphas[t,:] = local_alpha 
        
        logZ = sum_log([alphas[-1,s] for s in range(self.S)])    
        return alphas, logZ
                 
    def beta_chain(self, X, theta):
        
        T = X.shape[0]
        
        betas = np.zeros((T, self.S))
        for t in reversed(range(T)):
#                 if t+1 == T:
#                     betas[t,:] = [0.0 for k in range(self.S)]
            if t+1 < T:
                # For each state j: consider all possible previous states
                for i in range(self.S):
                    betas[t,i] = sum_log([betas[t+1,j] + self.log_psi(theta, t+1,i,j,X[t+1]) for j in range(self.S)])
                    
        return betas
           
    def expect(self, X, theta):   
        
        """
        Calculate tau=(gamma, xi) for each example and for each class (i.e. p(z_{k}|x_{i})) 
        and log-likelihood (i.e. \sum_{i}^{N} \ln \sum_{k}^{K} \pi_{k}p(x_{i}|z_{k}))
        """
        
        # probabilities p(x_{t}|z_{k})
        T = X.shape[0]
               
        alphas, logZ = self.alpha_chain(X, theta)
        betas = self.beta_chain(X, theta)
                      
        def gamma(t):
            """
            Smoothing. 
            Calculates probability of being in state i on step t given all data points
            (i.e. returns p(z_{t}|x) for each state i)
            """
            #s = sum_log([alphas[t,i] + betas[t,i] for i in range(self.S)])
            return [np.exp(alphas[t,i] + betas[t,i] - logZ) for i in range(self.S)]
    
        def xi(t):
            """
            Calculates posterior probability of transition from state i to state j at step t
            """
            if t == 0:
                return None
            
            xit = np.zeros((self.S, self.S))
            #norm = sum_log([alphas[t-1,k] + sum_log([self.log_psi(theta, t,k,s,X[t]) + betas[t,s] for s in range(self.S)]) for k in range(self.S)]) 
            
            for j in range(self.S):
                for i in range(self.S):
                    xit[i,j] = np.exp(alphas[t-1,i] + self.log_psi(theta, t,i,j,X[t]) + betas[t,j] - logZ)
           #     print 'sum(xi(',t,':,',j,'))=', sum(xit[:,j])
            return xit
    
        
        
        gammas = [gamma(t) for t in range(T)]    
        xis = [xi(t) for t in range(T)]

        return (gammas, xis), logZ
    
   
    def gradient_impl(self, gamma, xi, X, Y, theta):
        
       # regularizer
#        sigma_sq = 100.0
        
        grad = np.zeros(len(self.f))
        prev_s = self.S
        
        for t,s in enumerate(Y):
            xit = xi[t]
            x = X[t].A1
            
            begin = t*self.t_sens
            end = begin+self.t_sens
            
            for k in range(begin,end):
                # empirical
                g = self.f[k](t,prev_s,s,x) 
                # expected
                if t == 0:
                    g -= sum([gamma[0][j]*self.f[k](t,self.S,j,x) for j in range(self.S)])
                else:
                    g -= sum([xit[i,j]*self.f[k](t,i,j,x) for j in range(self.S) for i in range(self.S)])
                
                grad[k] += g
                
            for k in range(len(self.f)-self.t_insens, len(self.f)):
                # empirical
                g = self.f[k](t,prev_s,s,x) 
                # expected
                if t == 0:
                    g -= sum([gamma[0][j]*self.f[k](t,self.S,j,x) for j in range(self.S)])
                else:
                    g -= sum([xit[i,j]*self.f[k](t,i,j,x) for j in range(self.S) for i in range(self.S)])
                
                grad[k] += g
                    
            prev_s = s 
        
        # regularizer 
#         for k in range(self.f):
#             grad[k] -=  (theta[k] / sigma_sq)   
#                 
        return grad
            
    def gradient(self, X, Y, theta):
        
        (gamma, xi), logZ = self.expect(X, theta)
        
        grad = self.gradient_impl(gamma, xi, X, Y, theta)
        
        print 'normgrad', np.linalg.norm(grad)   
        return grad
            
        
    def train(self, X, Y, theta, disp=1, factr=1e12): 
        
        def objective(params, *args):
            crf = args[0]
            X = args[1]
            Y = args[2]
            return -crf.log_likelihood(X,Y,params)
        
        def objective_grad(params, *args):
            crf = args[0]
            X = args[1]
            Y = args[2]
            return crf.gradient(X,Y,params)
        
        return optimize.fmin_l_bfgs_b(objective, theta, fprime=objective_grad , args=(self,X,Y), disp=disp, factr=factr)       
        
                   