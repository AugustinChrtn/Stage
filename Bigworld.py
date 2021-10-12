import numpy as np

class BigState:
    #This is the reproduction of a classic Gridworld with rewards on transitions
    def __init__(self):
        # Set information about the states
        self.height = 20
        self.width = 20
        self.grid = np.zeros((self.height, self.width))
        self.final_states = {(3,3,'UP'):10,(18,18,'UP'):100 }
        self.values=dict()
        for x in range(self.height): # Loop through all possible grid spaces, create sub-dictionary for each
            for y in range(self.width):
                self.values[(x,y)] = {'UP':-1, 'DOWN':-1, 'LEFT':-1, 'RIGHT':-1} # Transition cost        
        self.current_location = (0,0)     
        self.first_location=(0,0)
        for transition, reward in self.final_states.items():
            self.values[(transition[0],transition[1])][transition[2]]=reward
        # Set available actions
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
              
    def make_step(self, action):
        """Moves the agent in the specified direction. If agent is at a border, agent stays still
        but takes negative reward. Function returns the reward for the move and if the move is a terminal action."""
        last_location = self.current_location       
        #Movements possible are UP, DOWN, LEFT and RIGHT and the borders of the square cannot be crossed
        reward = self.values[(last_location[0],last_location[1])][action]
        if action == 'UP':
            if last_location[0] != 0:
                self.current_location = ( last_location[0] - 1, last_location[1])        
        elif action == 'DOWN':
            if last_location[0] != self.height - 1:
                self.current_location = ( last_location[0] + 1, last_location[1])
        elif action == 'LEFT':
            if last_location[1] != 0:
                self.current_location = ( last_location[0], last_location[1] - 1)
        elif action == 'RIGHT':
            if last_location[1] != self.width - 1:
                self.current_location = ( last_location[0], last_location[1] + 1)        
        return reward, (last_location[0],last_location[1],action) in self.final_states.keys()
