import numpy as np
from collections import defaultdict, deque

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