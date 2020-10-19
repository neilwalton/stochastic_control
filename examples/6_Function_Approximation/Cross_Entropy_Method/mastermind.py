''' Mastermind

Short python code for the board game Mastermind
    
    https://en.wikipedia.org/wiki/Mastermind_(board_game)

Description:

    4 colored pegs are chosen and placed 4 slots in order. 
    This is hidden from the player. There are 6 colours to choose from.
    The player places combinations of 4. As feedback the player recieves the number of correct colors 
    and, amoung the remaining pegs, the number of correct colours in the incorrect place.
    The player wins when the correct 4 colours are placed in the correct order.
'''
import numpy as np
import string

class Mastermind():
    def __init__(self,num_slots=4,num_colors=6,solution=None):
        self.num_colors = num_colors
        self.num_slots = num_slots
            
        self.colors = list(range(num_colors))
        
        if solution is None:
            self.solution = np.random.choice(self.colors,self.num_slots)
        else:
            self.solution = solution
            
    def step(self,action):
        feedback = {'guess': action, 'correct': 0, 'correct_color': 0}
        
        # correct and incorrect pegs
        incorrect = []
        remaining_colors = []
        for peg in range(self.num_slots):
            if action[peg]==self.solution[peg] :
                feedback['correct']+=1
            else:
                incorrect.append(peg)
                remaining_colors.append(action[peg])
            
        # correct peg but in the wrong place
        for peg in incorrect :
            color = self.solution[peg]
            if color in remaining_colors :
                feedback['correct_color']+=1
                remaining_colors.remove(color)
                
        done = True if feedback['correct'] == self.num_slots else False
                
        return feedback, done
    
    def reset(self):
        self.solution = np.random.choice(self.colors,self.num_slots)
    
        
        