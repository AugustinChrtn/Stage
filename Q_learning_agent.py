import numpy as np

class Q_Agent():
    def __init__(self, environment, epsilon=0.01, alpha=0.26, gamma=0.7):
        self.environment = environment
        self.q_table = dict()
        self.counter=dict()
        for x in range(environment.height):
            for y in range(environment.width):
                self.q_table[(x,y)] = {'UP':0, 'DOWN':0, 'LEFT':0, 'RIGHT':0}
                self.counter[(x,y)]={'UP':0, 'DOWN':0, 'LEFT':0, 'RIGHT':0}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        
    def choose_action(self):
        """Returns the optimal action from Q-Value table. Makes an exploratory random 
            action if a uniform random variable is lower than epsilon."""
        if np.random.uniform(0,1) < self.epsilon:
            action = np.random.choice(['UP','DOWN','LEFT','RIGHT'])
        else:
            q_values_of_state = self.q_table[self.environment.current_location]
            maxValue = max(q_values_of_state.values())
            action = np.random.choice([k for k, v in q_values_of_state.items() if v == maxValue])
        self.counter[self.environment.current_location][action]+=1
        return action
    
    def learn(self, old_state, reward, new_state, action):
        q_values_of_state = self.q_table[new_state]
        max_q_value_in_new_state = max(q_values_of_state.values())
        current_q_value = self.q_table[old_state][action]        
        self.q_table[old_state][action] = (1 - self.alpha) * current_q_value + self.alpha * (reward + self.gamma * max_q_value_in_new_state)