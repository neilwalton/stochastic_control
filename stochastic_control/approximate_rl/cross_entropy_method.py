import numpy as np

class CrossEntropyCategorical:
    ''' Cross Entropy Method for Catagorical Data
    
    Source: 
       "The Cross-Entropy Method" 
       Dirk Kroese and Reuven Rubinstein. Springer (2004)
       [Algorithm 4.2.1 with variance smoothing from Remark 5.2]
    
    
    Arguments:
        probs  -- Array of probability distributions 
        rho    -- Cross entropy rarity parameter. (float)
                  [between 0. and 1.].         
        alpha  -- Smoothing parameter. (float)
                  [between 0 and 1. Default is 1. (no-smoothing)]  
        beta  -- Variance Smoothing parameter (float)
        q     -- polynomial decrease of Variance Smoothing parameter (int/float)
                  [q=0 means no decrease]
    '''

    def __init__(self, p, rho, alpha=1.):
        self.probs = p 
        self.rho = rho
        self.alpha = alpha
        self.time = 0 # number of training steps
        
    def train(self,x,y):
        # get top proportion rho
        x = np.array(x)
        y = np.array(y)
        cutoff = int(self.rho * len(y))
        idx = y.argsort()[-1*cutoff:]
        x_data = x[idx]
        
        # update probs
        probs_new = np.zeros(len(self.probs))
        categories, counts = np.unique(x_data, return_counts=True)  
        probs_new[categories] = counts
        probs_new = probs_new/np.sum(probs_new)        
        self.probs = self.alpha*probs_new + (1-self.alpha)*self.probs   
        
        self.time = self.time+1
        
        return cutoff, x_data
        
    def act(self,N=1):
        if N == 1: 
            return np.random.choice(len(self.probs), p=self.probs)        
        else:
            return np.random.choice(len(self.probs), N, p=self.probs) 
    
class CrossEntropyGaussian:
    ''' Cross Entropy Method -- Gaussian Implementation
    
    Source: 
       "The Cross-Entropy Method" 
       Dirk Kroese and Reuven Rubinstein. Springer (2004)
       [Algorithm 4.2.1 with variance smoothing from Remark 5.2]
    
    
    Arguments:
        mu     -- Initial mean of Gaussian. (numpy.ndarray)
        sigma  -- Initial covariance matrix of Gaussian . (np-array)
        rho    -- Cross entropy rarity parameter. (float)
                  [between 0. and 1.].         
        alpha  -- Smoothing parameter. (float)
                  [between 0 and 1. Default is 1. (no-smoothing)]  
        beta  -- Variance Smoothing parameter (float)
        q     -- polynomial decrease of Variance Smoothing parameter (int/float)
                  [q=0 means no decrease]
    '''
    
    def __init__(self,mu,sigma,rho,alpha=1.,beta=0.99,q=1):
        self.mu = np.array(mu)
        self.sigma = np.array(sigma)
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.q = q
        self.time = 0 # number of training steps
        
    def train(self,x,y):
        # calculate top rho of data
        y = np.array(y)
        cutoff = int(self.rho * len(y))
        idx = y.argsort()[-1*cutoff:]
        x_data = x[idx]
        
        # update mean and variance
        mu_new = np.mean(x_data, axis = 0)
        sigma_new = np.cov(x_data.T)        
        self.mu = self.alpha*mu_new + (1-self.alpha)*self.mu
        self.sigma = self.beta*sigma_new + (1-self.beta)*self.sigma
        
        # update time dependent parameters
        self.time = self.time+1
        self.beta = self.beta - self.beta*(1-1/self.time)**self.q
        
    def act(self,N):
        return np.random.multivariate_normal(self.mu, N, self.sigma, N)


class CrossEntropyMethod:
    ''' Cross Entropy Method -- for Optimization
    
    Source: 
       "The Cross-Entropy Method" 
       Dirk Kroese and Reuven Rubinstein. Springer (2004)
       [Algorithm 4.2.1]
      
    Arguments:
        rand_dist     -- returns random sample from given parameters/weights (function) 
                         e.g. rand_dist(weights) returns rv with desired weights
        weight_update -- fits weights from data X (function)
                         e.g. weight_update(X) returns optimized parameters for numpy array X, 
                         (X has entries generate from rand_dist)
        weights       -- initial parameters of probabilty distribution
        rho           -- Cross entropy rarity parameter. (float)
                         [between 0. and 1.].         
        alpha         -- Smoothing parameter. (float)
                         [between 0 and 1. Default is 1. (no-smoothing)]  
    '''
    
    def __init__(self, rand_dist, weight_update, weights, rho, alpha=1.):
        self.rv_fn = rand_dist 
        self.w_fn = weight_update
        self.weights = weights
        self.rho = rho
        self.alpha = alpha
        self.time = 0 # number of training steps
        
    def train(self,x,y):
        # get top proportion rho
        y = np.array(y)
        cutoff = int(self.rho * len(y))
        idx = y.argsort()[-1*cutoff:]
        x_data = x[idx]
        
        # update weights
        weights_new = weight_update(x_data)      
        self.weights = self.alpha*weights_new + (1-self.alpha)*self.weights       
        
        self.time = self.time+1
        
    def act(self,N):
        return np.array([self.rv_fn(self.weights) for _ in range(N)])

        
        

    
