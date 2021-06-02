import numpy as np
import matplotlib.pyplot as plt
from Gridworld import State
from Kalman_learning_agent import Kalman_agent
from Kalman_learning_agent_parallels import Kalman_agent_parallels
from Q_learning_agent import Q_Agent
from Useful_functions import play, display, find_best_triplet_Kalman,find_best_triplet_Q_learning,prefered_direction,gradient_variance_max,find_best_five_Kalman_parallels
from Bigworld import BigState
from Random_world import Random_state
from Kalman_raffine import Kalman_raffine
"""environment = BigState()
#Each agent is set with optimized values
KA = Kalman_agent(environment,gamma=0.8,variance_ob=1,variance_tr=0.01)
QA=Q_Agent(environment,epsilon=0.01,alpha=0.26,gamma=1)
KAP=Kalman_agent_parallels(environment,gamma=0.8,variance_ob=1,variance_tr=0.01,gamma_curiosity=0.3,curiosity_factor=10)

reward_QA, counter_QA, value_table_QA=play(environment, QA, trials=1000, max_steps_per_episode=1000, learn=True)
reward_KA,counter_KA,table_mean_KA,table_variance_KA=play(environment, KA, trials=1000, max_steps_per_episode=1000, learn=True)
reward_KAP, counter_KAP, table_mean_KAP,table_variance_KAP,table_curiosity_KAP=play(environment, KAP, trials=1000, max_steps_per_episode=1000, learn=True)"""

rewards=[[]for i in range(8)]
for i in range(1):
    environment = BigState()
    QA=Q_Agent(environment,epsilon=0.01,alpha=0.26,gamma=1)
    KA= Kalman_agent(environment,gamma=1,variance_ob=1,variance_tr=10000)
    KAP=Kalman_agent_parallels(environment,gamma=1,variance_ob=400,variance_tr=20,gamma_curiosity=0.4, curiosity_factor=20)
    KAR=Kalman_raffine(environment,gamma=1,variance_ob=100,variance_tr=20,gamma_curiosity=0.4, curiosity_factor=1,v_ob_curiosity=50,v_tr_curiosity=10)
    reward_QA, counter_QA,value_table_QA=play(environment, QA, trials=1000, max_steps_per_episode=1000, learn=True)
    reward_KA, counter_KA,table_mean_KA,table_variance_KA=play(environment, KA, trials=1000, max_steps_per_episode=1000, learn=True)
    reward_KAP, counter_KAP, table_mean_KAP,table_variance_KAP,table_curiosity_KAP=play(environment, KAP, trials=1000, max_steps_per_episode=1000, learn=True)
    reward_KAR, counter_KAR, table_mean_KAR, table_variance_KAR, table_curiosity_KAR=play(environment, KAR, trials=1000, max_steps_per_episode=1000, learn=True)
    rewards[0].append(reward_QA[200:])
    rewards[1].append(reward_QA)
    rewards[2].append(reward_KA[200:])
    rewards[3].append(reward_KA)
    rewards[4].append(reward_KAP[200:])
    rewards[5].append(reward_KAP)
    rewards[6].append(reward_KAR[200:])
    rewards[7].append(reward_KAR)
    
mean_rewards=[np.mean(reward) for reward in rewards]
print("avg_reward_QA 200+ = " +str(mean_rewards[0])+", "+"avg_reward_total_QA = "+str(mean_rewards[1]))
print("avg_reward_KA 200+ = " +str(mean_rewards[2])+", "+ "avg_reward_total_KA = "+str(mean_rewards[3]))
print("avg_reward_KAP 200+ = " +str(mean_rewards[4])+", "+"avg_reward_total_KAP = "+str(mean_rewards[5]))
print("avg_reward_KAR 200+ = " +str(mean_rewards[6])+", "+"avg_reward_total_KAR = "+str(mean_rewards[7]))


#bestt,besttt=find_best_five_Kalman_parallels()
#print(besttt,bestt[besttt])



plt.figure()
plt.plot(reward_QA, color='black',)
plt.xlabel("Numéro de l'essai (Q-learning)")
plt.legend()


plt.figure()
plt.plot(reward_KA, color='black')
plt.xlabel("Essai KA")
plt.legend()

plt.figure()
plt.plot(reward_KAP, color='black')
plt.xlabel("Essai KAP")
plt.legend()

plt.figure()
plt.plot(reward_KAR, color='black')
plt.xlabel("Essai KAR")
plt.legend()

