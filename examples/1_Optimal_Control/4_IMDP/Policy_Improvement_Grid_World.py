import numpy as np
from numpy.linalg import inv, det

Grid=np.matrix([0,0,0,-1,0,-2,0,0,0,0,0,+2])
r = np.transpose(Grid)
Epsilon = 0.8
beta = 0.9

P_Left = np.array(
                    [
                    [1,0,0,0,0,0,0,0,0,0,0,0],
                    [1,0,0,0,0,0,0,0,0,0,0,0],
                    [0,1,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,1,0,0,0,0,0,0,0,0],
                    [0,0,0,0,1,0,0,0,0,0,0,0],
                    [0,0,0,0,0,1,0,0,0,0,0,0],
                    [0,0,0,0,0,0,1,0,0,0,0,0],
                    [0,0,0,0,0,0,1,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,1,0,0,0],
                    [0,0,0,0,0,0,0,0,1,0,0,0],
                    [0,0,0,0,0,0,0,0,0,1,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,1]
                    ]
                   )

P_Right = np.array(
                    [
                    [0,1,0,0,0,0,0,0,0,0,0,0],
                    [0,0,1,0,0,0,0,0,0,0,0,0],
                    [0,0,0,1,0,0,0,0,0,0,0,0],
                    [0,0,0,1,0,0,0,0,0,0,0,0],
                    [0,0,0,0,1,0,0,0,0,0,0,0],
                    [0,0,0,0,0,1,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,1,0,0,0,0],
                    [0,0,0,0,0,0,0,1,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,1,0,0],
                    [0,0,0,0,0,0,0,0,0,0,1,0],
                    [0,0,0,0,0,0,0,0,0,0,0,1],
                    [0,0,0,0,0,0,0,0,0,0,0,1]
                    ]
                   )
                   
P_Up = np.array(
                    [
                    [0,0,0,0,1,0,0,0,0,0,0,0],
                    [0,1,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,1,0,0,0,0,0],
                    [0,0,0,1,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,1,0,0,0],
                    [0,0,0,0,0,1,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,1,0],
                    [0,0,0,0,0,0,0,0,0,0,0,1],
                    [0,0,0,0,0,0,0,0,1,0,0,0],
                    [0,0,0,0,0,0,0,0,0,1,0,0],
                    [0,0,0,0,0,0,0,0,0,0,1,0],
                    [0,0,0,0,0,0,0,0,0,0,0,1],
                    ]
                   )
                   
P_Down = np.array(
                    [
                    [1,0,0,0,0,0,0,0,0,0,0,0],
                    [0,1,0,0,0,0,0,0,0,0,0,0],
                    [0,0,1,0,0,0,0,0,0,0,0,0],
                    [0,0,0,1,0,0,0,0,0,0,0,0],
                    [1,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,1,0,0,0,0,0,0],
                    [0,0,1,0,0,0,0,0,0,0,0,0],
                    [0,0,0,1,0,0,0,0,0,0,0,0],
                    [0,0,0,0,1,0,0,0,0,0,0,0],
                    [0,0,0,0,0,1,0,0,0,0,0,0],
                    [0,0,0,0,0,0,1,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,1]
                    ]
                   )

P_Random = 0.25*P_Left + 0.25*P_Right + 0.25*P_Up + 0.25*P_Down

Q_Left = ( ( 1 - Epsilon ) * P_Left + Epsilon * P_Random ) 
Q_Right = ( ( 1 - Epsilon ) * P_Right + Epsilon * P_Random ) 
Q_Up = ( ( 1 - Epsilon ) * P_Up + Epsilon * P_Random ) 
Q_Down = ( ( 1 - Epsilon ) * P_Down + Epsilon * P_Random ) 

P = [Q_Left, Q_Right, Q_Up, Q_Down] # 0 = Left, 1 = Right, 2 = Up, 3 = Down

def Construct_Rewards_For_Transitions(pi,P,beta,r):
    Q_pi=[]
    I=np.identity(len(pi))
    
    for i in range(len(pi)):
        Q_pi.append(P[pi[i]][i])
    Q_pi=np.matrix(Q_pi)
    
    R_pi = inv( I - beta * Q_pi) @ r
    
    return R_pi

def Find_New_Policy(P,R_pi):
    
    new_pi=[]
    Reward_Left = beta * ( np.matrix(P[0]) @ R_pi )
    Reward_Right = beta * ( np.matrix(P[1]) @ R_pi )
    Reward_Up = beta * ( np.matrix(P[2]) @ R_pi )
    Reward_Down = beta * ( np.matrix(P[3]) @ R_pi )
    
    for state in range(len(pi)):
        action=np.argmax(
                [np.matrix.item(Reward_Left[state]),
                 np.matrix.item(Reward_Right[state]),
                 np.matrix.item(Reward_Up[state]),
                 np.matrix.item(Reward_Down[state])])
        new_pi.append(action)
    
    return new_pi

pi=[0,0,0,1,0,0,0,3,0,3,0,0]

for _ in range(100):
    R_pi = Construct_Rewards_For_Transitions(pi,P,beta,r)
    new_pi = Find_New_Policy(P,R_pi)
    print(new_pi)
    if new_pi == pi :
        break
    else:
        pi = new_pi
    

print(np.transpose(R_pi))