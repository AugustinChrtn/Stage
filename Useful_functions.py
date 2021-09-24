import numpy as np
import seaborn as sns
import copy

from Gridworld import State
from Bigworld import BigState

from Q_learning import Q_Agent
from Kalman import Kalman_agent
from Kalman_parallels import Kalman_agent_parallels
from Kalman_raffine import Kalman_agent_raffine
from Kalman_sum import Kalman_agent_sum


#Main play function
def play(environment, agent, trials=1000, max_steps_per_episode=1000, learn=True,photos=[0,10,50,100,200,300,500,800,999]):
    reward_per_episode,result_every_photo = [] , []  
    for trial in range(trials):
        if trial in photos:
            if type(agent).__name__=='Q_Agent': 
                value=copy.deepcopy(agent.q_table)
                result_every_photo.append(value)
            if type(agent).__name__ in ['Kalman_agent','Kalman_agent_raffine','Kalman_agent_parallels','Kalman_agent_sum','Kalman_agent_delayed_sum']: 
                value=copy.deepcopy(agent.KF_table_mean)
                result_every_photo.append(value)
        cumulative_reward, step, game_over= 0,0,False
        while step < max_steps_per_episode and game_over != True:
            old_state = environment.current_location
            action = agent.choose_action() 
            reward , terminal = environment.make_step(action)
            new_state = environment.current_location            
            if learn == True: # Update  if learning is specified
                agent.learn(old_state, reward, new_state, action)                
            cumulative_reward += reward
            step += 1            
            if terminal == True :
                environment.__init__()
                game_over = True                     
        reward_per_episode.append(cumulative_reward)
    if type(agent).__name__=='Q_Agent': return reward_per_episode, agent.counter, agent.q_table,result_every_photo
    if type(agent).__name__=='Kalman_agent': return reward_per_episode, agent.counter, agent.KF_table_mean, agent.KF_table_variance,result_every_photo
    if type(agent).__name__=='Kalman_agent_parallels': return reward_per_episode,agent.counter, agent.KF_table_mean,agent.KF_table_variance,agent.KF_table_curiosity, result_every_photo
    if type(agent).__name__=='Kalman_agent_raffine': return reward_per_episode,agent.counter, agent.KF_table_mean,agent.KF_table_variance,agent.KF_table_curiosity, result_every_photo
    if type(agent).__name__=='Kalman_agent_sum': return reward_per_episode,agent.counter,agent.KF_table_mean,agent.KF_table_variance, result_every_photo
    if type(agent).__name__=='Kalman_agent_delayed_sum': return reward_per_episode,agent.counter,agent.KF_table_mean,agent.KF_table_variance, agent.KF_table_curiosity, result_every_photo

#Search for the parameters that optimize the reward for each model after a given trial
def find_best_triplet_Kalman():
    best_triplet=dict()
    for gamma in range(100,101,5):
        for variance_ob in [10**i for i in range(0,5)]:
            for variance_tr in [i*10**4 for i in range(1,5)]:
                environment = BigState()
                KA= Kalman_agent(environment,gamma/100,variance_ob,variance_tr)
                reward_per_episode, counter_KA, table_mean, table_variance,result_every_photo_KA= play(environment, KA, trials=1000, learn=True)
                best_triplet["gamma = "+str(gamma/100), "variance_ob = "+str(variance_ob),"variance_tr "+str(variance_tr)]=np.mean(reward_per_episode[100:])
                v=list(best_triplet.values())
                best = list(best_triplet.keys())[v.index(max(v))]
    return best_triplet, best

def find_best_triplet_Q_learning():
    best_triplet=dict()
    for epsilon in range(1,30,3):
        for alpha in range(1,51,5):
            for gamma in range(5,11):
                environment = BigState()
                QA= Q_Agent(environment,epsilon/100,alpha/100,gamma/10)
                reward_per_episode, counter_QA, table, result_every_photo_QA= play(environment,QA, trials=1000, learn=True)
                best_triplet["epsilon="+str(epsilon/100), "alpha="+str(alpha/100),"gamma="+str(gamma/10)]=np.mean(reward_per_episode[200:])
                v=list(best_triplet.values())
                best = list(best_triplet.keys())[v.index(max(v))]
    return best_triplet, best   

def find_best_five_Kalman_raffine():
    best_five=dict()
    s=0
    for gamma in range(100,101,5):
        for variance_ob in [10**i for i in range(-1,4)]:
            for variance_tr in [10**i for i in range(1,4)]:
                for gamma_curiosity in range(1,11,3):
                    for curiosity_factor in [10**i for i in range(-2,3)]:
                        s+=1
                        if s%10==0:
                            print(s)
                        environment = BigState()
                        KAR= Kalman_agent_raffine(environment,gamma/100,variance_ob,variance_tr,gamma_curiosity/10,curiosity_factor)
                        reward_per_episode, counter_KAR,table_mean, table_variance,table_curiosity= play(environment, KAR, trials=1000, learn=True)
                        best_five["gamma = "+str(gamma/100), "variance_ob = "+str(variance_ob),"variance_tr = "+str(variance_tr),"gamma_curiosity = "+str(gamma_curiosity/10), "curiosity_factor = "+str(curiosity_factor)]=np.mean(reward_per_episode[500:])
                        dic_values=list(best_five.values())
                        best_value = list(best_five.keys())[dic_values.index(max(dic_values))]
    return (best_five,best_value)

def find_best_seven_Kalman_parallels():
    best_seven=dict()
    s=0
    for gamma in range(100,101,10):
        for variance_ob in [10**i for i in range(2,6)]:
            for variance_tr in [10**i for i in range(2,6)]:
                for gamma_curiosity in range(10,11,5):
                    for curiosity_factor in [10**i for i in range(-3,3)]:
                        for v_ob_curiosity in [10**i for i in range(3)]:
                            for v_tr_curiosity in [10**i for i in range(3)]:
                                s+=1
                                if s%10==0:
                                    print(s)
                                environment = BigState()
                                KAP= Kalman_agent_parallels(environment,gamma/100,variance_ob,variance_tr,gamma_curiosity/10,curiosity_factor,v_ob_curiosity,v_tr_curiosity)
                                reward_per_episode, counter_KAP,table_mean, table_variance,table_curiosity, result_every_photo_KAP= play(environment, KAP, trials=1000, learn=True)
                                best_seven["gamma = "+str(gamma/100), "variance_ob = "+str(variance_ob),"var    iance_tr = "+str(variance_tr),"gamma_curiosity = "+str(gamma_curiosity/10), "curiosity_factor = "+str(curiosity_factor), "v_ob_curiosity = "+str(v_ob_curiosity),"v_tr_curiosity = "+str(v_tr_curiosity)]=np.mean(reward_per_episode[500:])
                                dic_values=list(best_seven.values())
                                best_value = list(best_seven.keys())[dic_values.index(max(dic_values))]
    return (best_seven,best_value)

def find_best_quatuor_Kalman_sum():
    best_quatuor=dict()
    s=0
    for gamma in range(100,101,5):
        for variance_ob in [10**i for i in range(-1,4)]:
            for variance_tr in [10**i for i in range(1,4)]:
                    for curiosity_factor in [10**i for i in range(-2,3)]:
                        s+=1
                        if s%10==0:
                            print(s)
                        environment = BigState()
                        KAS= Kalman_agent_sum(environment,gamma/100,variance_ob,variance_tr,curiosity_factor)
                        reward_per_episode, counter_KAS,table_mean, table_variance, result_every_photo= play(environment, KAS, trials=1000, max_steps_per_episode=500, learn=True)
                        best_quatuor["gamma = "+str(gamma/100), "variance_ob = "+str(variance_ob),"variance_tr = "+str(variance_tr),"curiosity_factor = "+str(curiosity_factor)]=np.mean(reward_per_episode[500:])
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

def prefered_direction(table):
    decisions=[['NaN' for i in range(5)] for j in range(5)]
    for i in range(5):
        for j in range(5):
            v=list(table[i,j].values())
            best = list(table[i,j].keys())[v.index(max(v))]
            decisions[i][j]=best
    return decisions
        
def heatmap_variance(table_variance,environment):
     precision=np.array([[0 for i in range(environment.height)] for j in range(environment.width)])
     for i in range(environment.height):
         for j in range(environment.width):
             max_variance=max(table_variance[i,j].values()) 
             precision[i][j]=max_variance
     normalization_factor=np.max(precision)
     if normalization_factor !=0 : precision=precision/normalization_factor
     result=sns.heatmap(precision,cmap='Blues')
     return result
 
def max_dictionnary(d):
    maximum=max(d.values())


