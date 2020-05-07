import numpy as np
import random
from collections import defaultdict

class Q_function(defaultdict):
    '''
    Summary: 
        Stores the Q-function on an Markov Decision Process
        Is a dictionary sub-class
        
    Arguments:
        states 
            -- list of preset states
        actions 
            -- list of preset actions 
            -- these actions are assumed to exist for every state
                if this is not the case
                each state-action pair can be added with the self.add(state,action) method
              
    
    Usage:
        Q[state][action]
            -- gives value of state-action pair
        
        Q.max(state) 
            -- gives maximum value for state
            
        Q.argmax(state)
            -- gives the maximizing action
        
        Q.print()
            -- print the current Q-function
            
        Q.add(state), Q.add(state,action), Q.add(state,action,value)
            -- adds a state, action and value entry
    '''

    def __init__(self,states=None,actions=None):
        # get a dict inside each dictionary entry
        super(Q_function, self).__init__(dict)
        self.count = defaultdict(lambda : defaultdict())        
        ''' 
        returns the max / argmax / min / argmin value for a state 
        '''      
        
        # add preset states and actions
        if states is not None :
            for state in states :
                if actions is None :
                    self.add(state)
                else :
                    for action in actions :
                        self.add(state,action)
        
        if actions is not None :
            self.preset_actions = [self._hash(action) for action in actions ]  
        else :
            self.preset_actions = None

        
    def max(self,state):
        state = self._hash(state)
        not_empty = bool(self[state])
        return max(self[state].values()) if not_empty else 0.

    
    def argmax(self,state):
        state = self._hash(state)
        not_empty = bool(self[state])
        if not_empty :
            return max(self[state], key=self[state].get)
    
    def min(self,state):
        state = self._hash(state)
        not_empty = bool(self[state])
        if not_empty :
            return min(self[state], key=self[state].get)
    
    def argmin(self,state):
        state = self._hash(state)
        return min(self[state], key=self[state].get)
        
       

    def add(self,state,action=None,value=None):  
        ''' 
        add new states, actions, value 
        returns: hashed state and action
        '''
        state = self._hash(state)
        action = self._hash(action) if action is not None else None

        if state not in self.keys():
            self[state]
            if self.preset_actions is not None:
                self[state][action] = 0.
        if action is not None and action not in self[state].keys() :
            self[state][action] = 0.
            self.count[state][action] = 1
        if value is not None :
            self[state][action] = value
        
        return (state, action) if action is not None else state

    def print(self):
        '''
        prints each state, action and value
        '''
        for state in self.keys() :
            for action in self[state].keys():
                value = self[state][action]
                print(state,action,value)         
        
  
    def _hash(self,x):
        ''' 
        Hashes keys so can be added to a dictionary. E.g. x = Q._hash(x)
        '''
        if not isinstance(x, (int, float, str, type(None), np.number)):
            try:
                x = str(x)
            except:
                x = hash(x)
        return x
    

class Q_learning(Q_function):
    '''
    Summary: A Q_function with a Q-learning update
    
    works for maximizing rewards (currently)
    
    E.g. 
    Q.learn(state,action,reward,next_state)
    
    if no next_state then default exit state is None type
    '''
    def __init__(self,lr=0.1):
        self.lr = lr
        super(Q_learning, self).__init__()
        

    def learn(self,state,\
                  action,\
                  reward,\
                  next_state,\
                  done = False,\
                  discount=1.):   
        '''
        Q-learning update
        '''
        # add variables and hash where needed
        state, action = self.add(state,action)
        next_state = self.add(next_state)
        
        self.count[state][action] +=1
        
        # the direction of change
        
        dQ = reward \
            + discount * self.max(next_state) \
            - self[state][action]   
            
        # correct dQ if is end of episode
        if done or next_state is None:
            dQ = reward - self[state][action] 
    

            
                                     
        # The main Q-learning step
        self[state][action] = self[state][action] + self.lr * ( dQ )       
        
    def action(self,state,explore_prob=0.,actions=None):
        '''
        returns policy action for input state
        
        can randomize with explore_prob variable
        
        can add suggest list of actions (in case not seen before)
        '''
        
        if actions is not None :
            for act in actions :
                self.add(state,act)
        
        
        if random.random() > explore_prob :
            return self.argmax(state)
        else :
            random_action = random.choice(list(self[state].keys()))
            return random_action
    
        
        