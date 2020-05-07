'''
sequentially adds points works for upto five lines.
Example usage:
my_plot = rolling_plot(2)
for i in range(10):
    my_plot.plot(i,0)
    my_plot.plot(i**2,1)
my_plot.clear
'''
import matplotlib.pyplot as plt
from IPython import display

class rolling_plot():
    def __init__(self,n_indicies=1):
        self.n_indicies = n_indicies
        self.x_values = []
        self.colors = ['k','b','r','m','c'] 
        for _ in range(self.n_indicies):
            self.x_values.append([])        
    
    def plot(self,x,index=0):
        self.x_values[index].append(x)
        display.clear_output(wait=True)
        for idx in range(self.n_indicies):
            plt.plot(list(range(len(self.x_values[idx]))),self.x_values[idx], color=self.colors[idx])
        display.display(plt.gcf())
        plt.clf()
        
    def clear(self):
        self.x_values = []
        for _ in range(self.n_indicies):
            self.x_values.append([])
        plt.clf()