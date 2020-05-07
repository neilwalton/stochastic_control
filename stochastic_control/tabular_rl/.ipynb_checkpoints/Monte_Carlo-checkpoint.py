import numpy as np
from collections import defaultdict

class Monte_Carlo(defaultdict):
    """ Monte-carlo 
    
    Summary: Monte-Carlo estimation of value function
    
        N[s] <- N[s] + 1
        R[s] <- R[s] + (r-R[s])/N[s]
        
    # Arguments:
        discount - discount factor (float, default=1.)
        
    # Methods:
        .train(state,reward,done)
            - one iteration of Monte-Carlo (if done=True)
            
        .print()
            - prints a list of states reward estimates
    """
    
    def __init__(self,discount=1.):
        super(Monte_Carlo, self).__init__(float)
        self.disc = discount
        self.visits = defaultdict(int)
        self.ep_SR = [] # episode state-rewards
        
    def train(self,state,reward,done=False):   
        
        self.ep_SR.append((state,reward))
        
        # only update when episode is done
        if done or state is None: 
            # collate rewards 
            R = np.array([0])
            for s,r in reversed(self.ep_SR):
                R = np.append(r+self.disc*R[0],R)        
            
            # update average reward      
            for i, (s,r) in enumerate(self.ep_SR):
                self.visits[s] +=1
                self[s] +=  (R[i] -self[s])/self.visits[s]
            
            # reset for next episode        
            self.ep_SR = [] 
            
    def print(self):
        states = list(self.keys())
        try:
            states.sort()
        except:
            pass
            
        for state in states:
            print(state,self[state])