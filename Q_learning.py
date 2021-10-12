import numpy as np
import random

class Q_Agent():
    def __init__(self, environment, beta=0.05, alpha=0.95, gamma=1, epsilon=0.01, exploration='softmax'):
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
        self.exploration=exploration
        self.beta=beta
        
    def choose_action(self):
        """Returns the optimal action from Q-Value table. Makes an exploratory random 
            action if a uniform random variable is lower than epsilon."""
        if self.exploration=='e-greedy':
            if np.random.uniform(0,1) < self.epsilon:
                action = np.random.choice(['UP','DOWN','LEFT','RIGHT'])
            else:
                q_values_of_state = self.q_table[self.environment.current_location]
                maxValue = max(q_values_of_state.values())
                action = np.random.choice([k for k, v in q_values_of_state.items() if v == maxValue])
            return action
        if self.exploration=='softmax':
            q_values_of_state = self.q_table[self.environment.current_location]
            total=0
            for value in q_values_of_state.values():
                total+=np.exp(value*self.beta)
            for (key,value) in q_values_of_state.items():
                q_values_of_state[key]=np.exp(self.beta*value)/total
            action=random.choices(list(q_values_of_state.keys()),weights=q_values_of_state.values(),k=1)[0]
            return action
        self.counter[self.environment.current_location][action]+=1
        
    def learn(self, old_state, reward, new_state, action):
        q_values_of_state = self.q_table[new_state]
        max_q_value_in_new_state = max(q_values_of_state.values())
        current_q_value = self.q_table[old_state][action]        
        self.q_table[old_state][action] = (1 - self.alpha) * current_q_value + self.alpha * (reward + self.gamma * max_q_value_in_new_state)