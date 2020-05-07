class Value_Iteration():
    """ Value Iteration: A Finite Time Markov Decision Process Solver
    
    # Notes:
        Performs the update:
        
                    V <- max_a { r[x][a] +  E [ V[f[x][a]] | x,a ] }
            
        The code will also solve dynamic programs when dynamic is deterministic.
        Initial Values are zero by default, but can be editted.
    
    # Arguments 
        dynamic - 3D array e.g. P[action][state][next_state] for probability of next_state given action and state
        reward - 2D array giving reward from a state and action
        
        minimize (optional) - set true if you want to minimize rather than maximize
        states (optional) - array to specify all states
        actions (optional) - array to specify all actions
    
    # Example
    '''    
    # define transition probabilities
    dynamic = {
    'l':np.matrix([
            [0,0.5,0.5,0,0,0],
            [0,0,0,0,0,1.],
            [0,0,0,0,0,1.],
            [0,0,0,0,0,1.],
            [0,0,0,0,0,1.],
            [0,0,0,0,0,1.],
        ]) , 
    'r': np.matrix([
            [0,0,0,0.75,0.25,0],
            [0,0,0,0,0,1.],
            [0,0,0,0,0,1.],
            [0,0,0,0,0,1.],
            [0,0,0,0,0,1.],
            [0.,0,0,0,0,1.],
     ])
    }
    
    # define rewards 
    reward ={
    'l' :np.matrix([
        [0,4.,4.,0,0,0],
        [0,0,0,0,0,2.],
        [0,0,0,0,0,3.],        
        [0,0,0,0,0,6.],    
        [0,0,0,0,0,1.],  
        [0,0,0,0,0,0],  
    ]),    
    'r':np.matrix([
        [0,0,0,2.,2.,0],
        [0,0,0,0,0,2.],
        [0,0,0,0,0,3.],        
        [0,0,0,0,0,6.], 
        [0,0,0,0,0,1.],  
        [0,0,0,0,0,0],                 
    ])
    }
    
    # setup dp and solve over two time steps
    mdp = MDP(dynamic,reward,states=[0,1,2,3,4,5],actions=['l','r'])
    mdp.solve(2,0)
    '''
    
    
    # TO DO
    
    allow for array, dict and function input  
    """ 
    
    def __init__(self, dynamic, reward, minimize=False, states=None, actions=None):
        
        self.dynamic = dynamic
        self.reward = reward 
        
        
        
        self.value = 
        self.q_factor = 
        
    def train(self,time,state):
        V = np.zeros(len(states))
        for t in range(time):
            V_new = np.zeros(len(states))
            for state in states:
                Q = [ self.Q_Value(state,action,V) for action in actions]
                V_new[state] = np.max(Q)
            V = V_new
        return V
    
    def act(self,state):
        
    def Expectation(self,state,action,value_fn):
        E = self.dynamic[action][state] @ np.transpose(value_fn)
        return E.item(0)
    
    def Q_Value(self,state,action,value_fn):
        Q = self.Expectation(state,action,reward[action][state])\
            + self.Expectation(state,action,value_fn)
        return Q
    