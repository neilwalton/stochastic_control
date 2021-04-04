import numpy as np

class Value_Iteration:
    """ Value Iteration:
    
    # Summary 
    To solve finite time MDPs, performs value iteration step :
    
        V <- max_a { E [ r(x,a,x') + B * V[x'] | x,a] }

    # Arguments 
        P - Transition Probabilities 
            3D numpy array e.g. dynamic[action][state][next_state] 
            for probability of next_state given action and state
            
        r - rewards 
            2D numpy array e.g. reward[action][state][next_state]
            for reward from state and action
        
        disc - Discount factor.
               float (default = 1.). 
        
    # Methods
        .train()      - One iteration of value iteration. Returns value function
        .train(time)  - 'time' (int) iterations of value iteration
        .act(state)   - Returns recommended action for state (int)
        .policy()     - Returns policy (array)
                     
    # Example
        '''python
    
            dynamic = np.array([np.matrix([[0.5,0.5],[1.,0]]), np.matrix([[0.25,0.75],[1.,0]])])
            reward = np.array([[1.,-1.],[5.,-1.]])
            discount = 0.8

            v_iter = Value_Iteration(dynamic,reward,discount)
            v_iter.train(10)
            v_iter.policy()
        '''
    """   
    def __init__(self, nS, nA, P, disc=1.):
        # define problem parameters
        self.nA = nA
        self.nS = nS
        self.P = P
        self.disc = disc
        
        
        # set value function and Q-factors
        self.V = np.zeros(nS)
        self.Q = np.array([np.zeros(nA) for s in range(nS)]) 
    
    def train(self,iters=1,verbose=False):
        # for each iteration
        # for states, actions
        # find Q-factors as an expectation
        # take the maximum Q-factor as value
        for t in range(iters):
            for s in range(self.nS):
                for a in range(self.nA):
                    self.Q[s][a] = 0
                    for p,ns,r,d in self.P[s][a] :
                        if not d :
                            self.Q[s][a] += p * ( r + self.disc * self.V[ns] )
                        else: 
                            self.Q[s][a] += p * r
                            
            self.V = np.max(self.Q , axis=1)
            if verbose :
                print(self.V)
        
        return self.V
    
    def act(self,state):
        return np.argmax(self.Q[state])
    
    
class Policy_Iteration:
    """ Policy Iteration - a numerical solution to a MDP
    
    # Arguments 
        dynamic - 3D numpy array e.g. dynamic[action][state][next_state] 
                    for probability of next_state given action and state
        reward - 2D numpy array e.g. reward[action][state][next_state]
                    for reward from state and action
        discount - float (default = 1.). Discount factor.
    
    # Returns:
        policy from **one** policy iteration
        value function of input policy
        
    # Methods
        .train()      - One iteration of value iteration. Returns value function
        .train(time)  - 'time' (int) iterations of value iteration
        .act(state)   - Returns recommended action for state (int)
        .policy()     - Returns policy (array)
                     
    # Example
        '''python
    
            dynamic = np.array([np.matrix([[0.5,0.5],[1.,0]]), np.matrix([[0.25,0.75],[1.,0]])])
            reward = np.array([[1.,-1.],[5.,-1.]])
            discount = 0.8

            v_iter = Policy_Iteration(dynamic,reward,discount)
            v_iter.train(10)
            v_iter.policy()
        '''
    """

    def __init__(self, nS, nA, P, disc=.9):
        # define problem parameters
        self.nA = nA
        self.nS = nS
        self.P = P
        self.disc = disc
        
        # set value function and Q-factors
        self.V = np.zeros(nS)
        self.Q = np.array([np.zeros(nA) for _ in range(nS)]) 
        self.pi = np.zeros(nS,dtype=int)
    
    def train(self,time=1):
        for t in range(time):
            # Get rewards of policy pi
            self.V = self._Rewards()

            # 2nd of two main steps update Q-factors:
            for a in range(self.nA):
                for s in range(self.nS):   
                    self.Q[s][a] =0.
                    for p, ns, r, d in self.P[s][a]:
                        self.Q[s][a] += p * ( r + self.disc * self.V[ns])

            # policy iteration update 
            new_pi = np.argmax(self.Q, axis=1)
            
            if np.array_equal(new_pi,self.pi) :
                print('policy is optimal')
                break
            else :
                self.pi = new_pi
    
        return self.V

    def act(self,state):
        return self.pi[state]
    
    def _Rewards(self):
        # Output: Reward function of current policy
        
        # transitions and rewards of current policy
        self.P_pi = np.zeros((self.nS,self.nS))
        self.r_pi = np.zeros(self.nS)
        for s in range(self.nS):
            for p,ns,r,d in self.P[s][self.pi[s]]:
                self.P_pi[s][ns] += p
                self.r_pi[s] += p*r
        
        # Solve for Reward function        
        I = np.identity(self.nS)
        
        self.R_pi = np.linalg.solve(I - self.disc * self.P_pi, self.r_pi)

        return self.R_pi