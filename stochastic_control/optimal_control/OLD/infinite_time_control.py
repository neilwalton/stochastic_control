class DP():
    ''' Solves a dynamic program over finite time

    # Arguments:
        dynamic - 
        rewards - reward function
        

    # Output: 
        Value function at time and state
    '''
    if time > 0 :
        Q = [ r[state][action] + DP(time-1,f[state][action]) for action in A ]
        V = max(Q) 
    else :
        Q = r[state]
        V = max(Q) 
    return V

    def __init__(self,dynamic,reward): 
        self.dynamic = dynamic
        self.reward = reward 
        
    def train(self,time):
        pass
        
    def act(self):
        pass

def Policy_Iteration(pi,P,r,discount):
    ''' Policy Iteration - a numerical solution to a MDP
    
    # Arguments: 
        P - P[a][x][y] gives probablity of x -> y for action a 
        r - r[a][x][y] gives reward for x -> y for action a
        pi - pi[x] gives action for state x
        discount - disount factor
    
    # Returns:
        policy from **one** policy iteration
        value function of input policy
    '''
    
    # Collate array of states and actions
    number_of_actions, number_of_states = len(P), len(P[0])
    Actions, States = np.arange(number_of_actions), np.arange(number_of_states)
    
    # Get transitions and rewards of policy pi
    P_pi = np.array([P[pi[x]][x] for x in States ])
    r_pi = np.array([r[pi[x]][x] for x in States])
    Er_pi = [ np.dot(P_pi[x],r_pi[x]) for x in States]
    
    # Calculate Value of pi
    I = np.identity(number_of_states)
    A = I - discount * P_pi
    R_pi = np.linalg.solve(A, Er_pi)
    
    # Calculate Q_factors of pi
    Q = np.zeros((number_of_actions,number_of_states))
    for a in range(number_of_actions):
        for x in range(number_of_states):          
            Q[a][x] = np.dot(P[a][x],r[a][x]+discount*R_pi) 
    
    # policy iteration update 
    pi_new = np.argmax(Q, axis=0)
    
    return pi_new, R_pi

def Value_Iteration(V,P,r,discount):
    ''' Value Iteration - a numerical solution to a MDP
    
    # Arguments: 
        V - a 1D np.array. V[x] gives value for state x
        P - a 3D np.array. P[a][x][y] gives probablity of x -> y for action a 
        r - a 3D np.array. r[a][x][y] gives reward for x -> y for action a
        discount - a float. disount factor
    
    # Returns:
        Value function and policy from **one** value iteration
    '''
    number_of_actions = len(P)
    number_of_states = len(P[0])
    
    Q = np.zeros((number_of_actions,number_of_states))
    
    for a in range(number_of_actions):
        for x in range(number_of_states):          
            Q[a][x] = np.dot(P[a][x],r[a][x]+discount*V) 
            
    V_new = np.amax(Q, axis=0)
    
    pi_new = np.argmax(Q, axis=0)
    
    return V_new, pi_new