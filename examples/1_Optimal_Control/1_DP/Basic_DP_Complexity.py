import numpy as np
import random 

import time 
import matplotlib.pyplot as plt

problem_size = 25 # 20 is a better number if you don't like waiting 

def Get_Data(problem_size):
    data=[]
    for i in range(problem_size):
        data.append(np.random.randint(9, size=2**i))
    return data

def Simple_DP(data):
    problem_size=len(data)
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

x=np.zeros(len(data))
y=np.zeros(len(data))

for i in range(problem_size):
    start_time=time.time()
    output=Simple_DP(data[0:i])
    x[i]=i
    y[i]=time.time()-start_time
    print('run time of problem', i,' is ', time.time()-start_time)

plt.plot(x, y, color='lightblue', linewidth=3)
plt.xlim(0, problem_size)
plt.xlabel('Tree Depth')
plt.ylabel('Run Time (seconds)')
plt.show()

plt.plot(x, np.log(y), color='lightblue', linewidth=3)
plt.xlim(0, problem_size)
plt.xlabel('Tree Depth')
plt.ylabel('Run Time (seconds)')
plt.show()