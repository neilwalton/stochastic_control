import numpy as np
from tqdm import tqdm

def brownian_motion(time=1.,accuracy=2**-8, levels=None):
    ''' Brownian motion
    
    # Summary 
    Simulation of Brownian motion (continuous process with independent normally distributed increments)
    
    # Arguments
        time      - time length (float)
        accuracy  - smallest increment size (float)
        levels    - a list of accuaries multiplied to the smallest increment 
                    an increasing list of integers starting with 1 e.g. [1, 2, 4, 8] 
                    
    # Returns
        If levels is None, returns a numpy array of times and numpy of the path
        If levels set, returns a two dictionaries of times and paths with levels as keys

    # Example 
        >> # returns a list of times and a brownian path
        >> times, path = brownian_motion() 
        >>
        >> # returns a dictionary of times and a dictionary of paths for the same brownian motion
        >> # accuracies for levels=[1,4,16] are [2**-8,2**-6,2**-4]
        >> t_list, p_list = brownian_motion(accuracy=2**-8,levels=[1,4,16])  
    
    '''
    
    # determine finest level
    if levels is not None:
        acc = levels[0]*accuracy
    else :
        acc = accuracy
    
    N = int(time // acc)
    DBM = np.random.normal(0,1,N)
    # Sum up Brownian increments at the fine level
    times = [0.]
    BM = [0.]
    for t, DB in enumerate(DBM):
        times.append((t+1)*acc)
        BM.append(BM[-1]+DB) #if len(B)> 0 else B=[DB]
    BM=np.array(BM)/np.sqrt(N)
    times=np.array(times)
    
    # If multiple levels specified return a dictionary with the brownian motion
    if levels is None :    
        return times, BM
    else :
        BM_dict = dict()
        times_dict = dict()
        for l in levels :
            gap = l // levels[0] 
            coarse_times = [i for i in range(0,len(BM),gap)]
            BM_dict[l] = BM[coarse_times]
            times_dict[l] = times[coarse_times]
        return times_dict, BM_dict
    
class Euler_Maruyama:
    ''' Euler Maruyama Scheme
    
    # Summary 
    Implementation of Euler Maruyama scheme for simulating Stochastic Differential Equations

        dX = mu(X) dt + sigma(X) dB  

    where mu is drift, sigma is volility, dt is time increment, dB is Brownian increment
    
    # Arguments
        mu        -- drift function (function)
        sigma     -- volatility function (function)
        x0        -- start point
        time      -- time interval
        
    # Methods
        .sim(times,BM)      -- simulate one sample path at given list of times
                             - times (np.array) increasing list of times
                             - BM (np.array) a list of driving BM values
        .sim_levels(levels) -- simulate paths for each level with same Brownian motion
                             - levels (array int) 
                             - times_dict (dict) dictionary of times for driving BM
                             - BM_dict (dict) dictionary of values for driving BM
    
    '''
    
    def __init__(self,
                 mu,
                 sigma,
                 x0=0.,
                 time=1.,
                 accuracy=2**-8):
        
        self.mu = mu
        self.sigma = sigma
        self.x0 = x0
        self.time = time
        self.accuracy = accuracy
    
    def sim(self,times,BM=None):
        x=self.x0
        Path=[x]
        for i, t in enumerate(times):
            if i == 0:
                continue
            
            D = t-times[i-1]
            DW = np.random.normal(0,np.sqrt(D)) if BM is None else BM[i]-BM[i-1]
                           
            x += self.mu(x) * D + self.sigma(x) * DW
            Path.append(x)
        
        return Path     
    
    def sim_levels(self,levels,times_dict=None,BM_dict=None):
        
        if BM_dict is None :        
            times_dict, BM_dict = brownian_motion(self.time,
                                                  self.accuracy, 
                                                  levels=levels)
            
        
        Paths_dict = dict()
        for key in times_dict.keys():
            Path = self.sim(times_dict[key],BM_dict[key])
            Paths_dict[key]=Path
            
        return times_dict, Paths_dict
    
class MLMC:
    
    def __init__(self,levels,iterations,method,objective):
        ''' Multi-Level Monte-Carlo 
        
        # Arguments      
            levels     -- Level and number of samples at that level. (dict)
                          Highest accuracy level first.
            iterations -- Number of iterations at each level. 
            method     -- Numerical method e.g. Euler Maruyama, Milstein.
                          The method requires method.sim_levels(levels).
            objective  -- Python function taking a path and evaluating a property. (function)
            
        # Methods
            .run()     -- runs MLMC for required number of iterations.
                          returns an MLMC estimate. (float)      
                          
        # Example 
            >> EM = Euler_Maruyama(lambda x: 1,lambda x : 1, x0=1., accuracy=2**-6)
            >> levels = [1, 4, 16]
            >> iterations = [4, 10, 20]
            >> objective = lambda path : path[-1]
            >> mlmc=MLMC(levels,iterations,EM,objective)
            >> estimate = mlmc.run()
            >> print(estimate) # real answer is 2.
        '''
                  
        self.levels = levels
        self.iterations = iterations
        self.method = method
        self.obj_fn = objective
        
        
    def run(self):
        obj = np.array([0. for l in self.levels])
        idx = 0
        N=1
        pbar = tqdm(total=np.sum(self.iterations[-1]))
        mlmc_estimate = 0.
        
        for i, N in enumerate(self.iterations) :
            for _ in range(N):       
                pbar.update(1)
                if i == len(self.iterations)-1:
                    times_dict, Paths_dict \
                    = self.method.sim_levels(levels=[self.levels[i]])
                    P = self.obj_fn(Paths_dict[self.levels[i]])         
                    mlmc_estimate += P/N

                else :
                    times_dict, Paths_dict \
                    = self.method.sim_levels(levels=[self.levels[i], self.levels[i+1]])
                    P = self.obj_fn(Paths_dict[self.levels[i]])  
                    P_minus = self.obj_fn(Paths_dict[self.levels[i+1]])
                    mlmc_estimate += (P - P_minus)/N
        
        pbar.close()
                
        return mlmc_estimate               
        
            
def mlmc_iterations(epsilon,costs,variances):
    ''' Multi-level calculator
    
    Number of iterations at each level required by MLMC for RMS epsilon for given costs and variances 
    '''
    C_sq = np.sqrt(costs)
    V_sq = np.sqrt(variances)
    
    N = np.divide(V_sq,C_sq) * np.dot(V_sq,C_sq) / epsilon**2
    
    return np.ceil(N).astype(int), np.dot(N,costs)
                   
        
    
