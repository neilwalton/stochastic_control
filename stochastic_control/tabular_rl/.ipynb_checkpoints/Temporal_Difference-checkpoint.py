import numpy as np
from collections import defaultdict, deque


class TD_Learn(defaultdict):
    """ TD-learning 
    
    Summary: performs the TD(0) update:
    
        V[x] <- V[x] + lr * ( r + B V[x'] - V[x] )
    
    # Arguments 
        lr       - learning rate (float).      
        discount - disount factor (float)
        states   - list of states. (default is empty)
        
    # Methods
        .train(state,reward,next_state,done,lr)   
            - One iteration of TD(0).
           
    # Additional Methods
        .copy()         - returns a new copy of self
        .print()        - prints current Q-function   
                     
    # Example
        '''python
        
        V = TD_Learn()
        
        state = env.reset()      
        done = False
        
        while done is False:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            V.train(state,reward,next_state,done)
            state = next_state
        '''
    """
    def __init__(self,lr=0.01,discount=1.,states=None):
        super(TD_Learn, self).__init__(float)
        self.lr = lr
        self.disc = discount
        
        if states is not None:
            for state in states:
                    self[state]
                        
    def train(self,state,reward,next_state,done=False,lr=None):     
        # get temporal difference then update 
        if lr is not None:
            self.lr = lr
        
        if not done: 
            TD = reward + self.disc * self[next_state] - self[state] 
        else : 
            TD = reward - self[state]
           
        self[state] += self.lr * TD    
    
    def print(self):
        states = list(self.keys())
        try:
            states.sort()
        except:
            pass
            
        for state in states:
            print(state,self[state])
                
    def copy(self,lr=None):
        lr = self.lr if lr is None else lr
        V_copy = TD_Learn(lr=lr,discount=self.disc)
        
        for state in self.keys():
            V_copy[state] = self[state]
        
        return V_copys
    
    
class TD_n_step(defaultdict):
    """ n-step TD 
    
    Summary: performs the n-step update:
    
        V[x0] <- V[x0] + lr * ( r1 + ... + rn + V[xn] - V[x0])
               = V[x0] + lr * ( td1 + ... + tdn )
    
    # Arguments 
        steps    - number of steps before update
        lr       - learning rate (float).      
        discount - disount factor (float)
        states   - list of states. (default is empty)
        
    # Methods
        .train(state,reward,next_state,done,lr)   
            - One iteration of TD(0).
           
    # Additional Methods
        .copy()         - returns a new copy of self
        .print()        - prints current Q-function   
                     
    # Example
        '''python
        
        V = TD_n_step(4)
        
        state = env.reset()      
        done = False
        
        while done is False:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            V.train(state,reward,next_state,done)
            state = next_state
        '''
    """
    def __init__(self,steps,lr=0.01,discount=1.):
        super(TD_n_step, self).__init__()
        self.steps = steps
        self.lr = lr
        self.disc = discount
        self.n_SRD = deque([(None,0.,True) for _ in range(steps)],maxlen=steps)
                        
    def train(self,state,reward,done):            
        (s0,r0,d0)=self.n_SRD.popleft() 
        self.n_SRD.append(state,reward,done)
        
        TD_sum = r0 - self[s0]
        V_n = self.disc**self.steps * self[state][action] 
        for idx, (s,r,d) in enumerate(self.n_SRD) :
            TD_sum += self.disc**(idx+1) * r
            if d:
                V_n = 0.
                break
        TD_sum+=V_n
        
        self[s0][a0]+= self.lr * TD_sum
        
        return TD_sum
    
    def print(self):
        states = list(self.keys())
        try:
            states.sort()
        except:
            pass
            
        for state in states:
            print(state,self[state])
                
    def copy(self,lr=None):
        lr = self.lr if lr is None else lr
        V_copy = TD_Learn(lr=lr,discount=self.disc)
        
        for state in self.keys():
            V_copy[state] = self[state]
        
        return V_copy  
    
    
class TD_Lambda(defaultdict):
    """ TD-Lambda
    
    Summary: Performs the TD(lambda) (online version with eligability traces)
    
        E[x'] <- (lambda * B) * E[x'] + lr * I[x'=x]
        V[x'] <- V[x'] + E[x'] * TD
    
    # Arguments 
        lr       - learning rate (float).      
        discount - disount factor (float)
        states   - list of states. (default is empty)
        
    # Methods
        .train(state,reward,next_state,done,lr)   
            - One iteration of TD(lambda).
           
    # Additional Methods
        .copy()         - returns a new copy of self
        .print()        - prints current Q-function   
                     
    # Example
        '''python
        
        V = TD_Learn()
        
        state = env.reset()      
        done = False
        
        while done is False:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            V.train(state,reward,next_state,done)
            state = next_state
        '''
    """
    def __init__(self,Lambda,lr=0.01,discount=1.,states=None):
        super(TD_Lambda, self).__init__()
        self.El = defaultdict() # Eligibility Trace
        self.lr = lr
        self.disc = discount
        self.lamb = Lambda
        
        if states is not None:
            for state in states:
                    self[state]
                    self.El[state]
                        
    def train(self,state,reward,next_state,done=False,lr=None):     
        # get temporal difference then update 
        if lr is not None:
            self.lr = lr
        
        if not done: 
            TD = reward + self.disc * self[next_state] - self[state] 
        else : 
            TD = reward - self[state]
           
        # update eligability trace
        for x in self.El.keys():
            self.El[x] *= self.lamb * self.disc
        self.El[state] += self.lr
        
        for x in self.keys():
            self[x] += self.El[x]*TD
    
    def print(self):
        states = list(self.keys())
        try:
            states.sort()
        except:
            pass
            
        for state in states:
            print(state,self[state])
                
    def copy(self,lr=None):
        lr = self.lr if lr is None else lr
        V_copy = TD_Lambda(Lambda=self.lamb,lr=lr,discount=self.disc)
        
        for state in self.keys():
            V_copy[state] = self[state]
        
        return V_copy
    