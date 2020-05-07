import numpy as np
import random 

problem_size = 3

def Get_Data(problem_size):
    data=[]
    for i in range(problem_size):
        data.append(np.random.randint(9, size=2**i))
    return data

def Get_Data(problem_size):
    data=[]
    for i in range(problem_size):
        data.append(np.random.randint(9, size=2**i))
    return data

def Simple_DP(data):
    for i in range(problem_size-1,0,-1):
        for j in range(0, len(data[i-1])):
            '''The next line is the main dynamic programing step'''
            data[i-1][j]+=min(data[i][2*j],data[i][2*j+1])
            arg = np.argmin([data[i][2*j],data[i][2*j+1]])
            if arg == 0:
                data[i][2*j]=1 
                data[i][2*j+1]=0
            else: 
                data[i][2*j]=0
                data[i][2*j+1]=1
    return data

data=Get_Data(problem_size)
print(data)

output=Simple_DP(data)
print(output)

print('Cost is:', data[0][0])