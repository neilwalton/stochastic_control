import numpy as np
import random
from collections import defaultdict, deque



'''
##################################
##       MC & TD METHODS        ##
##################################
'''


# class Monte_Carlo(defaultdict):
#     """ Monte-carlo 
    
#     Summary: Simple Monte-Carlo estimation of reward function
    
#     """
    
#     def __init__(self,discount=1.,states=None):
#         super(TD_Learn, self).__init__()
#         self.disc = discount
#         self.visits = defaultdict(int)
#         self.ep_SR = [] # episode state-rewards
        
#         if states is not None:
#             for state in states:
#                     self[state]
#                     self.visits[state]

#     def train(self,state,reward,done=False):     

#         self.ep_SR.append((state,reward))
        
#         if done or state is None: 
#             # collate rewards add all states and actions
#             R = np.array([])
#             for s,r in reversed(self.ep_SR):
#                 R = np.append(r,self.disc*R)
#                 self[s][a]               
            
#             # update average reward      
#             for i, (s,r) in enumerate(self.ep_SAR):
#                 self.visits[s] +=1
#                 self[s] +=  (R[i] -self[s])/self.visits[s]
                    
#             self.ep_SR = [] # reset for next episode

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
        super(TD_Learn, self).__init__(defaultdict(float))
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
        
        return V_copy

    
# class TD_n_step(defaultdict):
#     """ n-step TD 
    
#     Summary: performs the n-step update:
    
#         V[x0] <- V[x0] + lr * ( r1 + ... + rn + V[xn] - V[x0])
#                = V[x0] + lr * ( td1 + ... + tdn )
    
#     # Arguments 
#         steps    - number of steps before update
#         lr       - learning rate (float).      
#         discount - disount factor (float)
#         states   - list of states. (default is empty)
        
#     # Methods
#         .train(state,reward,next_state,done,lr)   
#             - One iteration of TD(0).
           
#     # Additional Methods
#         .copy()         - returns a new copy of self
#         .print()        - prints current Q-function   
                     
#     # Example
#         '''python
        
#         V = TD_n_step(4)
        
#         state = env.reset()      
#         done = False
        
#         while done is False:
#             action = env.action_space.sample()
#             next_state, reward, done, _ = env.step(action)
#             V.train(state,reward,next_state,done)
#             state = next_state
#         '''
#     """
#     def __init__(self,steps,lr=0.01,discount=1.,states=None):
#         super(TD_n_step, self).__init__()
#         self.steps = steps
#         self.lr = lr
#         self.disc = discount
        
#         # maintain a list of last n discounts, (cummulative) TDs, and states
#         self.discs = [ discount**(steps-i) for i in range(steps)]
#         self.TDs = np.zeros(steps)
#         self.n_states = [ None for _ in range(steps)]
        
#         # add given states
#         if states is not None:
#             for state in states:
#                     self[state]
                        
#     def train(self,state,reward,next_state,done=False,lr=None):     
#         # get temporal difference then update 
#         if lr is not None:
#             self.lr = lr
        
#         if not done: 
#             new_TD = reward + self.disc * self[next_state] - self[state] 
#         else : 
#             new_TD = reward - self[state]
            
#         self.TDs += self.discs * new_TD
#         self.TDs = np.append(self.TDs,new_TD)
#         self.n_states.append(state)
             
#         if not done:
#             # just update the state with all n TDs
#             nth_state = self.n_states[0]    
#             if nth_state is not None:
#                 self[nth_state] += self.lr * self.TDs[0]  
                
#             self.n_states = self.n_states[1:]
#             self.TDs = self.TDs[1:]            
#         else:
#             # update all states with existing TDs and reset
#             for idx, x in enumerate(self.n_states):
#                 if x is not None:
#                     self[x] += self.lr * self.TDs[idx]
                    
#             self.n_states = [ None for _ in range(steps)]
#             self.TDs = np.zeros(steps)
    
#     def print(self):
#         states = list(self.keys())
#         try:
#             states.sort()
#         except:
#             pass
            
#         for state in states:
#             print(state,self[state])
                
#     def copy(self,lr=None):
#         lr = self.lr if lr is None else lr
#         V_copy = TD_Learn(lr=lr,discount=self.disc)
        
#         for state in self.keys():
#             V_copy[state] = self[state]
        
#         return V_copy
    
    
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
    
'''
##########################################
##     Q Learning and SARSA methods     ##
##########################################
'''
    
    
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
    
'''
########################################################
##      Policy Gradients and Actor-Critic Methods     ##
########################################################
'''

class REINFORCE_tabular(defaultdict):
    ''' REINFORCE implementation for tabular RL
    
    Summary: 
        stores log-odds for each state-action pair, hence is tabular.
        Uses Monte-Carlo baseline.
    
        update:
        
        theta[x][a] <- theta + lr * (R - B[x]) * d log pi(a|x,theta)
        
        for softmax tabular distribution:
    
        pi(a|x,theta) = exp{ theta[x][a] } / (sum_{a'} exp{ theta[x][a'] })
        
        d_{a'} log pi(a|x,theta) = I_{a,a'} - pi(a'|x,theta)
        
        baseline is monte-carlo (above: Monte_Carlo).
                        
    Reference:  
        Simple statistical gradient-following algorithms for connectionist reinforcement learning. 
        Williams, R. J. (1992). Machine learning, 8(3-4), 229-256.    
    '''   
    
    def __init__(self,lr=0.01,discount=1.,states=None,actions=None):
        super(REINFORCE_tabular, self).__init__(lambda: defaultdict(float))
        self.lr = lr
        self.disc = discount
        self.actions = actions
        self.ep_SAR = []
        self.baseline = Monte_Carlo()
        
        if states is not None and actions is not None:
            for state in states:
                for action in actions:
                    self[state][action]
                    self.pi[state][action] = 1/len(actions)
                    
    def train(self,state,action,reward,done=False,lr=None):     
        # It's a monte-carlo method so only trains when episode finished        
        if lr is not None:
            self.lr = lr
            
        self.ep_SAR.append((state,action,reward))
        self.baseline.train(state,reward,done)
        
        if done or state is None: 
            # collate rewards add all states and actions
            R = np.array([])
            for s,a,r in reversed(self.ep_SAR):
                R = np.append(r,self.disc*R)
                self[s][a]               
            self._update_pi()
            
            # update theta parameters (self[s][a])            
            for i, (s,a,r) in enumerate(self.ep_SAR):
                self[s][a] += self.lr * (R[i]-self.baseline[s])
                for b in self[s].keys():
                    self[s][b] -= self.lr * (R[i]-self.baseline[s]) * self.pi[s][b]                    
            self._update_pi()            
            self.ep_SR = [] # reset for next episode
            
                        
    def act(self,state,actions=[]):    
        if actions :
            for a in actions :
                self[state][a]
                
        self._update_pi()               
        keys = self.pi[state].keys()
        probs = np.array(self.pi[state].values())
          
        return np.random.choice( keys, p=probs )
               
    def _update_pi(self):
        # calculates policy for current theta (self[s][a])
        for s in self.keys():
            Z = sum(self[s].values())
            for a in self[s].keys():
                self.pi[s][a] = self[s][a]/Z
                
class Actor_Critic_tabular:
    """ Actor Critic implmentation for tabular RL 
    
    Summary: 
        Actor    - (self.actor) stores log-odds for each state-action pair, hence is tabular.
                 - applies soft-max over actions for each state
        Critic   - (self.critic) n-step TD critic
        
        update:
        
        TODO need to send back TDs (useful for Actor-Critic Later)
    
    """   
    def __init__(self,steps=1,lr_actor=0.01,lr_critic=0.01,discount=1.):
        self.lr_actor = lr_actor
        self.disc = discount
        self.steps = steps
        self.SARDs = deque([(None,None,0.,True) for _ in range(steps)],maxlen=steps)
        self.actor = defaultdict(lambda: defaultdict(float))
        self.pi = defaultdict(lambda: defaultdict(float))
        self.critic = TD_n_step(steps,lr_critic)   
        
    def train(self,state,action,reward,done):                                
        TD = self.critic.train(state,reward,done)
        
        (s0,a0,r0,d0) = self.SARDs.popleft()
        self.SARDs.append(state,action,reward,done)     
                
        for b in self[s0].keys():
            self.actor[s0][b] -= self.lr_actor * TD * self.pi[s0][b]               
        self.actor[s0][a] += self.lr_actor * TD
        
        self._update_pi(s0)   
        
    def act(self,state,actions=[]):    
        if actions :
            for a in actions :
                self[state][a]
                
        self._update_pi()               
        keys = self.pi[state].keys()
        probs = np.array(self.pi[state].values())
          
        return np.random.choice( keys, p=probs )
        
    def _update_pi(self,state):
        Z = sum(self[state].values())
        for a in self[state].keys():
            self.pi[state][a] = self[state][a]/Z

                    
