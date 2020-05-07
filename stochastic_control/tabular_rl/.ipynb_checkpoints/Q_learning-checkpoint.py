class Q_Learn(defaultdict):
    """ Q-learning 
    
    Summary: performs the Q-learning update:
    
        Q[x][a] <- Q[x][a] + lr * ( r[x][a] + B max_a' Q[x'][a'] - Q[x][a] ) 
    
    # Arguments 
        lr       - learning rate (float).      
        discount - disount factor (float)
        states   - list of states. (default is empty)
        actions  - list of actions. (default is empty)
                   actions are assumed for all states
        
    # Methods
        .train(state,action,reward,next_state,done,lr)   
            - One iteration of Q-learning.
            
        .act(state,explore,actions)   
            - returns action for state with given explore probability
           
    # Additional Methods
        .max(state)     - maximizing value
        .argmax(state)  - maximizing action
        .policy()       - returns python function for policy
        .copy()         - returns a new copy of self
        .print()        - prints current Q-function
                     
    # Example
        '''python
        
        Q = Q_Learn()
        
        state = env.reset()      
        done = False
        
        while done is False:
            action = Q.act(state)
            next_state, reward, done, _ = env.step(action)
            Q.train(state,action,reward,next_state,done)
            state = next_state

        '''
    """
    def __init__(self,lr=0.01,discount=1.,states=None,actions=None):
        super(Q_Learn, self).__init__(lambda: defaultdict(float))
        self.lr = lr
        self.disc = discount
        self.actions = actions
        if states is not None and actions is not None:
            for state in states:
                for action in actions:
                    self[state][action]
                        
    def train(self,state,action,reward,next_state,done=False,lr=None):     
        # get temporal difference then update Q-factor
        if lr is not None:
            self.lr = lr
        
        dQ = reward + self.disc * self.max(next_state) - self[state][action]   
        if done or next_state is None:
            dQ = reward - self[state][action]                                             
        self[state][action] += self.lr * dQ       
        
    def act(self,state,explore=0.,actions=[]):           
        # exploit step with prob explore_prob (if state has seen an action)
        # explore step otherwise (amougst all actions if none raise an error)
        
        keys = list(self[state].keys())
        if random.random() > explore and bool(keys):
            # exploit:
            return self.argmax(state)
        
        else: 
            # explore:           
            # get all the actions together
            all_actions = []
            for lst in [keys, actions, self.actions]:
                if lst :
                    all_actions.extend(lst)
            all_actions = list(set(all_actions))
            
            # then, if non-empty, chose one at random
            if all_actions:
                random_action = random.choice(all_actions)
                return random_action    
            else:
                raise Exception("no actions given")
        
    def max(self,state):
        not_empty = bool(self[state])
        return max(self[state].values()) if not_empty else 0.
    
    def argmax(self,state):
        not_empty = bool(self[state])
        if not_empty :
            return max(self[state], key=self[state].get)
        
    def policy(self,explore=0.,actions=[]):
        return (lambda state : self.act(state,explore=explore,actions=actions))
    
    def print(self):
        states = list(self.keys())
        try:
            states.sort()
        except:
            pass
            
        for state in states:
            for action in self[state].keys():
                    print(state,action,self[state][action])
                
    def copy(self,lr=None):
        lr = self.lr if lr is None else lr
        Q_copy = Q_Learn(lr=lr,discount=self.disc,actions=self.actions)
    
        for state in self.keys():
                for action in self[state].keys():
                    Q_copy[state][action] = self[state][action]
        
        return Q_copy