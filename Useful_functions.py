import numpy as np
from Gridworld import State
from Kalman_learning_agent import Kalman_agent
from Kalman_learning_agent_parallels import Kalman_agent_parallels
from Q_learning_agent import Q_Agent


#Main play function
def play(environment, agent, trials=500, max_steps_per_episode=1000, learn=True):
    reward_per_episode = []   
    for trial in range(trials):
        cumulative_reward = 0
        step = 0
        game_over = False
        while step < max_steps_per_episode and game_over != True: # Run until max steps or until game is finished
            old_state = environment.current_location
            action = agent.choose_action() 
            reward , terminal = environment.make_step(action)
            new_state = environment.current_location            
            if learn == True: # Update  if learning is specified
                agent.learn(old_state, reward, new_state, action)                
            cumulative_reward += reward
            step += 1            
            if terminal == True : # If game is in terminal state, game over and start next trial
                environment.__init__()
                game_over = True                     
        reward_per_episode.append(cumulative_reward)
    if type(agent).__name__=='Q_Agent': return reward_per_episode, agent.q_table
    if type(agent).__name__=='Kalman_agent': return reward_per_episode, agent.KF_table_mean, agent.KF_table_variance
    if type(agent).__name__=='Kalman_agent_parallels': return reward_per_episode,agent.KF_table_mean,agent.KF_table_variance,agent.KF_table_curiosity


#Search for the parameters that optimize the reward for each model after a given trial
def find_best_triplet_Kalman():
    best_triplet=dict()
    for gamma in range(80,101,5):
        for variance_ob in [1/(10**i) for i in range(-2,2)]:
            for variance_tr in [1/(10**i) for i in range(-2,2)]:
                environment = State()
                KA= Kalman_agent(environment,gamma/100,variance_ob,variance_tr)
                reward_per_episode, table_mean, table_variance= play(environment, KA, trials=1000, learn=True)
                best_triplet["gamma = "+str(gamma/100), "variance_ob = "+str(variance_ob),"variance_tr "+str(variance_tr)]=np.mean(reward_per_episode[500:])
                v=list(best_triplet.values())
                best = list(best_triplet.keys())[v.index(max(v))]
    return best_triplet, best

def find_best_triplet_Q_learning():
    best_triplet=dict()
    for epsilon in range(1,11,2):
        for alpha in range(1,51,5):
            for gamma in range(5,11):
                environment = State()
                QA= Q_Agent(environment,epsilon/100,alpha/100,gamma/10)
                reward_per_episode, table= play(environment,QA, trials=1000, learn=True)
                best_triplet["epsilon="+str(epsilon/100), "alpha="+str(alpha/100),"gamma="+str(gamma/10)]=np.mean(reward_per_episode[500:])
                v=list(best_triplet.values())
                best = list(best_triplet.keys())[v.index(max(v))]
    return best_triplet, best   

def find_best_quatuor_Kalman_parallels():
    best_quatuor=dict()
    for gamma in range(70,96,5):
        for variance_ob in [1/(10**i) for i in range(-2,0)]:
            for variance_tr in [1/(10**i) for i in range(0,2)]:
                for gamma_curiosity in range(1,5):
                    environment = State()
                    KAP= Kalman_agent_parallels(environment,gamma/100,variance_ob,variance_tr,gamma_curiosity/10)
                    reward_per_episode, table_mean, table_variance,table_curiosity= play(environment, KAP, trials=1000, learn=True)
                    best_quatuor["gamma = "+str(gamma/100), "variance_ob = "+str(variance_ob),"variance_tr "+str(variance_tr),"gamma_curiosity = "+str(gamma_curiosity/10)]=np.mean(reward_per_episode[500:])
                    dic_values=list(best_quatuor.values())
                    best_value = list(best_quatuor.keys())[dic_values.index(max(dic_values))]
    return (best_quatuor,best_value)



#Visualisation functions
def display(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            display(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))

def chosen_direction(table):
    decisions=[['NaN' for i in range(5)] for j in range(5)]
    for i in range(5):
        for j in range(5):
            v=list(table[i,j].values())
            best = list(table[i,j].keys())[v.index(max(v))]
            decisions[i][j]=best
    return decisions
        
def gradient_variance_min(table_variance):
     precision=[[0 for i in range(5)] for j in range(5)]
     for i in range(5):
         for j in range(5):
             min_variance=min(table_variance[i,j].values()) 
             precision[i][j]=min_variance
     return precision

