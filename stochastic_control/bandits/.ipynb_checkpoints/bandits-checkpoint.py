import numpy as np
from numpy.random import beta

class Bernoulli_arm:
    ''' Bernoulli distributed arm
    
    Argument:
        p -- probability of success 
    
    Method:
        pull() -- pulls the arm reward 1. or 0.
    
    '''
    def __init__(self,p):
        self.prob = p
        self.num_pulls = 0
        self.tot_reward = 0.
        
    def pull(self):
        self.num_pulls += 1
        if np.random.rand() < self.prob :
            self.tot_reward += 1.
            return 1.
        else :
            return 0.
        
class Bernoulli_multiarm(list):
    ''' Bernoulli Multi-arm
    
    Summary: An array of Bernoulli arms
    
    Argument:
        p -- an array of probabilities for each arm
    
    Methods:
        rewards() -- array with total reward for each arm       
        pulls()   -- array with number of pulls of each arm   
    '''
    
    def __init__(self,p):
        super().__init__()
        self.probs = p
        for prob in p :
            self.append(Bernoulli_arm(prob))
            
    def rewards(self):
        return [arm.tot_reward for arm in self]
    
    def pulls(self):
        return [arm.num_pulls for arm in self]
        
class epsilon_greedy_bandit:
    ''' Epsilon Geedy Bandit Algorithm
    
    Summary: 
        chooses arm with maximum reward 
        else, with probability epsilon, chooses a random arm
        
    Arguments:
        num_actions   -- number of actions/arms (int)
        explore_prob  -- exploration probability (float)
        
    Additional Arguments:
        decay    -- if decay probability (bool) 
                    decay function is min(epsilon/time,1)
        decay_fn -- custom decay function
                    (python function one variable [time] as input)
    
    Methods: 
        act()             -- returns arm index (int)
        train(arm,reward) -- add to rewards of arm        
    
    Example:
        >>> bandit = epsilon_greedy_bandit(10,0.1)
        >>> arm = bandit.act()
        >>> reward = Arms[arm].pull()
        >>> bandit.train(arm,reward)
    '''
    
    def __init__(self, num_actions, epsilon, decay=False, decay_fn=None):  
        self.num_arms = num_actions
        self.rewards = [ 0. for arm in range(self.num_arms)] 
        self.time = 0 # amount of training data *not* number of plays        
        
        self.epsilon = epsilon      
        self.decay = decay
        self.decay_fn = decay_fn
        
    def train(self, arm, reward):
        self.rewards[arm] += reward
        self.time += 1
    
    def act(self):     
        prob = self._explore_prob(self.time,self.decay,self.decay_fn)
        if np.random.rand() < prob:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.rewards)
        
    def _explore_prob(self, time, decay, decay_fn):
        if decay is False:
            return self.epsilon
        elif decay_fn is None:
            return self.epsilon/(time+1)
        else:
            return decay_fn(time)
        
class exp3_bandit:
    ''' Exponential weights for Exploration and Exploitation (Exp3)
    
    Summary: 
        Apply exponential weight to loss [negative reward plus 1] of each arm
        sample proportional to distribution of weights
    
    Source: 
        Exp3 algorithm [with eta=sqrt(log K/tK)] page 24 of
        'Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems'
        Bubeck and Cesa-Bianchi. Foundations and Trends in Machine Learning (2012)
        
    Arguments:
        num_actions -- number of actions/arms (int)

    Methods: 
        act()             -- returns arm index (int)
        train(arm,reward) -- add to rewards of arm   
    '''
    
    def __init__(self, num_actions):  
        self.num_arms = num_actions
        self.time = 0 # amount of training data *not* number of plays    
        self.weights = np.ones(self.num_arms)
        self.probs = self.weights/np.sum(self.weights)
        self.losses = np.array([ 0. for arm in range(self.num_arms)]) # not cummulative loss, is reweighted by probs
        self.eta = 0.
        
    def train(self, arm, reward):
        self.time += 1
        self.losses[arm] += (1-reward)/self.probs[arm]
        self.eta = np.sqrt(np.log(self.num_arms)/(self.num_arms*self.time))
        self.weights = np.exp(-1.*self.eta*self.losses)
        self.probs = self.weights/np.sum(self.weights)
        
    def act(self):
        return np.random.choice(self.num_arms,p=self.probs)
    
class samba_bandit:
    ''' Stochastic Approximation Multi-arm Bandit Algorithm (SAMBA)
    
    Summary: 
        A stochastic approximation algorithm for multi-arm bandits.
        
    Arguments:
        num_actions -- number of actions/arms (int)
        
    Additional Arguments:
        alpha -- learning rate between 0 and 1 (float)
        cooling -- apply cooling to learning rate (bool)

    Methods: 
        act()             -- returns arm index (int)
        train(arm,reward) -- add to rewards of arm   
    '''
    
    def __init__(self, num_actions, alpha=0.1, cooling=False):
        self.num_arms = num_actions
        self.probs = np.ones(self.num_arms)/self.num_arms
        self.alpha = alpha
        self.cooling = cooling
        if self.cooling :
            self.alphas = self.alpha / (1-np.log(self.probs))
        else:
            self.alphas = self.alpha * np.ones(self.num_arms)
            
        
    def train(self,arm,reward):
        arm_star = np.argmax(self.probs)
        p_star = self.probs[arm_star]
        
        if arm == arm_star :
            self.probs *= ( 1 - self.alphas * (self.probs) * reward / p_star )
        else :
            self.probs[arm] += self.alphas[arm] * self.probs[arm] * reward
        self.probs[arm_star] = 1-np.sum(np.delete(self.probs, arm_star, 0))
        
        if self.cooling :
            self.alphas = self.alpha / (1-np.log(self.probs))
        
    def act(self):
        return np.random.choice(self.num_arms,p=self.probs)
    
class Thompson_bandit:
    ''' Thompson Sampling Bandit
    
    Summary:
        Maintains a beta distribution for each arm
        Samples one for each arm and recommends the maximal sample.
       
    Arguments:
        num_actions -- number of actions/arms (int)
    
    Additional Arguments
        alpha -- initial alpha for beta distribution (float)
        beta  -- initial beta for beta distribution (float)
        
    Methods: 
        act()             -- returns arm index (int)
        train(arm,reward) -- add to rewards of arm              
    '''
    
    def __init__(self,num_actions,alpha=1.,beta=1.):  
        self.num_arms = num_actions
        self.alphas = [alpha for _ in range(self.num_arms)]
        self.betas = [beta for _ in range(self.num_arms)]
        
    def train(self, arm, reward):
        self.alphas[arm] += reward
        self.betas[arm] += 1. - reward
    
    def act(self):
        sample = [ np.random.beta(self.alphas[arm], self.betas[arm]) \
                      for arm in range(self.num_arms)]
        return np.argmax(sample)
    
class UCB_bandit:
    ''' UCB Bandit Algorithm
    
    Summary:
        Chooses arm with maximal mean reward plus confidence bound.

    Source: UCB1 algorthim from
            'Finite-time Analysis of the Multiarmed Bandit Problem' 
            Auer, Cesa-Bianchi, Fischer. JMLR 2002.

    Arguments:
        num_actions -- number of actions/arms (int)

    Methods: 
        act()             -- returns arm index (int)
        train(arm,reward) -- add to rewards of arm    
    '''
    
    def __init__(self,num_actions):
        self.num_arms = num_actions
        self.time = 0
        self.trials = [0 for _ in range(self.num_arms)] # number trails per arm
        self.x_hat = [0. for _ in range(self.num_arms)] # mean reward per arm
        self.UCBs = [np.inf for _ in range(self.num_arms)] # upper-confidence bounds per arm
        
    def train(self, arm, reward):
        self.trials[arm] += 1
        self.time += 1
        self.x_hat[arm] += (reward - self.x_hat[arm])/self.trials[arm] 
            
    def act(self): 
        for arm in range(self.num_arms):
            self.UCBs[arm] = self.x_hat[arm] \
                            + ( 2*np.log(self.trials[arm])/self.time )**(1./2)\
                            if self.trials[arm] > 0 else np.inf
            
        return np.argmax(self.UCBs)
    
    
class Gradient_Bandit_Algorithm:
    ''' Gradient Bandit Algorithm 
    
    Summary:
        Applies gradient ascent to exponential weight for each arm
        
    Source: Section 2.8
        'Reinforcement Learning: An Introduction'
        Richard S. Sutton and Andrew G. Barto
        
    Arguments:
        num_actions -- number of actions/arms (int)
        alpha       -- learning rate (float)

    Methods: 
        act()             -- returns arm index (int)
        train(arm,reward) -- add to rewards of arm    
    
    '''
    def __init__(self,num_actions,alpha):
        self.num_arms = num_actions
        self.alpha = alpha
        self.time = 0
        self.av_reward = 0.
        self.weights = np.zeros(self.num_arms)
        self.probs =np.ones(self.num_arms)/self.num_arms
        
    def _update_probs(self):
        exp_weights = np.exp(self.weights)
        self.probs = exp_weights/sum(exp_weights)
        
    def train(self, arm, reward):
        self.time += 1
        self.av_reward += (reward - self.av_reward)/self.time
        self.weights -= self.alpha * (reward - self.av_reward) * self.probs
        self.weights[arm] += self.alpha * (reward - self.av_reward)
        self._update_probs()
        
    def act(self):
        return np.random.choice(self.num_arms,p=self.probs)
    
    
class samba_bandit_metropolis:
    ''' Stochastic Approximation Multi-arm Bandit Algorithm (SAMBA)
    
    Summary: 
        A stochastic approximation algorithm for multi-arm bandits.
        
    Arguments:
        num_actions -- number of actions/arms (int)
        
    Additional Arguments:
        alpha -- learning rate between 0 and 1 (float)
        cooling -- apply cooling to learning rate (bool)

    Methods: 
        act()             -- returns arm index (int)
        train(arm,reward) -- add to rewards of arm   
    '''
    
    def __init__(self, num_actions, alpha=0.1, cooling=False):
        self.num_arms = num_actions
        self.probs = np.ones(self.num_arms)/self.num_arms
        self.alpha = alpha
        self.cooling = cooling
        self.current_arm = 0
        if self.cooling :
            self.alphas = self.alpha / (1-np.log(self.probs))
        else:
            self.alphas = self.alpha * np.ones(self.num_arms)
            
        
    def train(self,arm,reward):
        arm_star = np.argmax(self.probs)
        p_star = self.probs[arm_star]
        
        if arm == arm_star :
            self.probs *= ( 1 - self.alphas * (self.probs) * reward / p_star )
        else :
            self.probs[arm] += self.alphas[arm] * self.probs[arm] * reward
        self.probs[arm_star] = 1-np.sum(np.delete(self.probs, arm_star, 0))
        
        if self.cooling :
            self.alphas = self.alpha / (1-np.log(self.probs))
        
    def act(self):
        next_arm = (self.current_arm + 1) % self.num_arms
        
        if np.random.rand() < self.probs[next_arm]/self.probs[self.current_arm]:
            self.current_arm = next_arm
        
        return self.current_arm
        