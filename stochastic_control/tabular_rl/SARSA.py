import numpy as np
from collections import defaultdict, deque

class SARSA(defaultdict):
    """ SARSA
    
    Summary: performs the Q-learning update:
    
        Q[x][a] <- Q[x][a] + lr * ( r[x][a] + B Q[x'][a'] - Q[x][a] ) 
    
    # Arguments 
        lr       - learning rate (float).      
        discount - disount factor (float)
        states   - list of states. (default is empty)
        actions  - list of actions. (default is empty)
                   these actions are assumed to exist for all states
        
    # Methods
        .train(state,action,reward,next_state,next_action,done,lr)   
            - One iteration of SARSA.
            
        .act(state,explore,actions)   
            - returns action for state with given explore probability
            - actions can be given (in addition to pre-existing actions) 
              (default is empty)
           
    # Additional Methods
        .max(state)     - maximizing value
        .argmax(state)  - maximizing action
        .policy()       - returns python function for policy
        .copy()         - returns a new copy of self
        .print()        - prints current Q-function
                     
    # Example
        '''python
        
        Q = SARSA()
        
        state = env.reset()      
        action = Q.act(state,explore=0.1)
        done = False
        
        while done is False:
            next_state, reward, done, _ = env.step(action)
            next_action = Q.act(next_state,explore=0.1)
            
            Q.train(state,action,reward,next_state,next_action,done)
           
            state = next_state
            action = next_action
        '''
    """
    def __init__(self,lr=0.01,discount=1.,states=None,actions=None):
        super(Q_Learn, self).__init__(lambda : defaultdict(float))
        self.lr = lr
        self.disc = discount
        self.actions = actions
        if states is not None and actions is not None:
            for state in states:
                for action in actions:
                    self[state][action]
                        
    def train(self,state,action,reward,next_state,next_action,done=False,lr=None):     
        # get temporal difference then update Q-factor
        if lr is not None:
            self.lr = lr
        
        td = reward + self.disc * self[next_state][next_action] - self[state][action]   
        if done or next_state is None:
            td = reward - self[state][action]                                             
        self[state][action] += self.lr * td       
        
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
    


class SARSA_n_step(defaultdict):
    """ n-step SARSA 
    
    Summary: performs the n-step update:
    
        Q[x0][a0] <- Q[x0][a0] + lr * ( r0 + ... + r{n} + Q[state][action] - Q[x0][a0] )
    
    # Arguments 
        lr       - learning rate (float).      
        discount - disount factor (float)
        states   - list of states (default is empty)
        actions  - list of action (default is empty)
                   these actions are assumed to exist for all states
        
    # Methods
        .train(state,action,reward,next_state,next_action,done,lr)   
            - One iteration of SARSA.
            
        .act(state,explore,actions)   
            - returns action for state with given explore probability
            - actions can be given (in addition to pre-existing actions) 
              (default is empty)
           
    # Additional Methods
        .max(state)     - maximizing value
        .argmax(state)  - maximizing action
        .policy()       - returns python function for policy
        .copy()         - returns a new copy of self
        .print()        - prints current Q-function
                     
    # Example
        '''python
        
        Q = SARSA()
        
        state = env.reset()      
        action = Q.act(state,explore=0.1)
        done = False
        
        while done is False:
            next_state, reward, done, _ = env.step(action)
            next_action = Q.act(next_state,explore=0.1)
            
            Q.train(state,action,reward,next_state,next_action,done)
           
            state = next_state
            action = next_action

        '''
    """
    def __init__(self,steps,lr=0.01,discount=1.,states=None,actions=None):
        super(SARSA_n_step, self).__init__(lambda: defaultdict(float))
        self.steps = steps
        self.lr = lr
        self.disc = discount
        self.actions = actions
        
        # maintain last n (state,action,reward,done)
        self.n_SARD = deque([ (None,None,0.,True) for _ in range(steps)],maxlen=steps)
                
        if states is not None and actions is not None:
            for state in states:
                for action in actions:
                    self[state][action]
                        
    def train(self,state,action,reward,done):     
        # update: one update from n-steps ago (which maybe from previous episode)
        # return: n-step temporal difference        
        (s0,a0,r0,d0) = self.n_SARD.popleft()   
        self.n_SARD.append((state,action,reward,done))          

        # sum rewards until n-steps or done 
        TD_sum = r0 - self[s0][a0]
        V_n = self.disc**self.steps * self[state][action] 
        for idx, (s,a,r,d) in enumerate(self.n_SARD) :
            TD_sum += self.disc**(idx+1) * r
            if d:
                V_n = 0.
                break
        TD_sum+=V_n
                
        return TD_sum

        
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
        Q_copy = SARSA_n_step(lr=lr,discount=self.disc,actions=self.actions)
    
        for state in self.keys():
                for action in self[state].keys():
                    Q_copy[state][action] = self[state][action]
        
        return Q_copy

    
class SARSA_Lambda(defaultdict):
    """ SARSA(lambda)
    
    Summary: performs the SARSA(lambda) update (online version with eligability traces)
    
        E[x'][a'] <- (lambda * B) * E[x'][a'] + lr * I[x'=x,a'=a]
        Q[x'][a'] <- Q[x'][a'] + E[x'][a'] * TD
    
    # Arguments 
        lr       - learning rate (float).      
        discount - disount factor (float)
        states   - list of states. (default is empty)
        actions  - list of actions. (default is empty)
                   these actions are assumed to exist for all states
        
    # Methods
        .train(state,action,reward,next_state,next_action,done,lr)   
            - One iteration of SARSA.
            
        .act(state,explore,actions)   
            - returns action for state with given explore probability
            - actions can be given (in addition to pre-existing actions) 
              (default is empty)
           
    # Additional Methods
        .max(state)     - maximizing value
        .argmax(state)  - maximizing action
        .policy()       - returns python function for policy
        .copy()         - returns a new copy of self
        .print()        - prints current Q-function
                     
    # Example
        '''python
        
        Q = SARSA()
        
        state = env.reset()      
        action = Q.act(state,explore=0.1)
        done = False
        
        while done is False:
            next_state, reward, done, _ = env.step(action)
            next_action = Q.act(next_state,explore=0.1)
            
            Q.train(state,action,reward,next_state,next_action,done)
           
            state = next_state
            action = next_action

        '''
    """
    def __init__(self,Lambda,lr=0.01,discount=1.,states=None,actions=None):
        super(SARSA_Lambda, self).__init__(lambda: defaultdict(float))
        self.lr = lr
        self.disc = discount
        self.actions = actions
        self.lamb = Lambda
        if states is not None and actions is not None:
            for state in states:
                for action in actions:
                    self[state][action]
                    
    def train(self,state,action,reward,next_state,next_action,done=False,lr=None):     
        # get temporal difference then update 
        if lr is not None:
            self.lr = lr
        
        if done or next_state is None: 
            TD = reward - self[state][action]
        else : 
            TD = reward + self.disc * self[next_state][next_action] - self[state][action]

        # update eligability trace
        # NSW: could be more efficient with just states from current epsiode
        for x in self.El.keys():
            for a in self.El[x].keys():
                self.El[x][a] *= self.lamb * self.disc
        self.El[state][action] += self.lr
        
        for x in self.keys():
            for a in self[x].keys():
                self[x][a] += self.El[x]*TD 
        
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
        Q_copy = SARSA_Lambda(lr=lr,discount=self.disc,actions=self.actions)
    
        for state in self.keys():
                for action in self[state].keys():
                    Q_copy[state][action] = self[state][action]
        
        return Q_copy
    
