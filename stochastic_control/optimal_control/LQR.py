import numpy as np

'''
Solve Riccati equation in various forms
'''

def riccati(L,A,B,C):
    ''' riccati equation solver
    
    # Summary
        One iteration of the riccati recursion for linear dynamic
            x' = A x + B a
        and cost
            (x,a) C (x,a)
        given value function at the next time is given by the matrix L
    
    # Arguments
        L,C -- np.matrix (should be semi-definite)
        A,B -- np.matrix
    '''

    dim_x = 