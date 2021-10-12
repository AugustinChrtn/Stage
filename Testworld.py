import numpy as np

class TestState:
    #This is the reproduction of a classic Gridworld with rewards on transitions
    def __init__(self):
        # Set information about the states
        self.height = 20
        self.width = 20
        self.transition_cost=1
        self.values=np.zeros((self.height,self.width))
        self.first_location=(0,0)
        self.current_location = (0,0)     
        self.good_finish= (4,4)
        self.best_finish = (19,19)
        self.terminal_locations = [self.good_finish,self.best_finish]
        self.values[self.good_finish] = 10
        self.values[self.best_finish]=100
        # Set available actions
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
              
    def make_step(self, action):
        """Moves the agent in the specified direction. If agent is at a border, agent stays still
        but takes negative reward. Function returns the reward for the move and if the move is a terminal action."""
        last_location = self.current_location       
        #Movements possible are UP, DOWN, LEFT and RIGHT and the borders of the square cannot be crossed
        if action == 'UP':
            if last_location[0] != 0 :
                self.current_location = (last_location[0]-1,last_location[1])        
        elif action == 'DOWN':
            if last_location[0] != self.height-1:
                self.current_location = (last_location[0]+1, last_location[1])
        elif action == 'LEFT':
            if last_location[1] != 0:
                self.current_location = (last_location[0], last_location[1]-1)
        elif action == 'RIGHT':
            if last_location[1] != self.width-1:
                self.current_location = (last_location[0], last_location[1]+1)
        reward = self.values[self.current_location]-self.transition_cost
        return reward, self.current_location in self.terminal_locations
