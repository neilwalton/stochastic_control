import numpy as np
from collections import defaultdict


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
