import numpy as np

class BigState:
    #This is the reproduction of a classic Gridworld with rewards on transitions
    def __init__(self):
        # Set information about the states
        self.height = 8
        self.width = 8
        self.grid = np.zeros((self.height, self.width))
        self.values=dict()
        for x in range(self.height): # Loop through all possible grid spaces, create sub-dictionary for each
            for y in range(self.width):
                self.values[(x,y)] = {'UP':-1, 'DOWN':-1, 'LEFT':-1, 'RIGHT':-1} # Transition cost        
        # Set random start location for the agent
        self.current_location = (0,0)       
        # Set the reward for a specific transition
        self.good_transition = (3,3,'UP')
        #self.loop_transition=(4,4,'UP')
        self.best_transition = (6,6,'UP')
        self.terminal_actions = [self.good_transition,self.best_transition]
        self.values[(self.good_transition[0],self.good_transition[1])][self.good_transition[2]] = 10
        #self.values[(self.loop_transition[0],self.loop_transition[1])][self.loop_transition[2]]=10
        self.values[(self.best_transition[0],self.best_transition[1])][self.best_transition[2]]=100
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
        return reward, (last_location[0],last_location[1],action) in self.terminal_actions
