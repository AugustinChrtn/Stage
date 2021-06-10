import numpy as np
import matplotlib.pyplot as plt
from Gridworld import State
from Kalman_learning_agent import Kalman_agent
from Kalman_learning_agent_parallels import Kalman_agent_parallels
from Q_learning_agent import Q_Agent
from Useful_functions import play, display, find_best_triplet_Kalman,find_best_triplet_Q_learning,prefered_direction,gradient_variance_max,find_best_five_Kalman_parallels, find_best_seven_Kalman_raffine
from Bigworld import BigState
from Random_world import Random_state
from Kalman_raffine import Kalman_agent_raffine
from softmax_variance import Kalman_agent_softmax
from Kalman_raffine_faux import Kalman_agent_raffine_faux
from Kalman_parallels_faux import Kalman_agent_parallels_faux

rewards=[[]for i in range(14)]
for i in range(10):
    environment = State()
    QA=Q_Agent(environment,epsilon=0.01,alpha=0.26,gamma=1)
    KA= Kalman_agent(environment,gamma=1,variance_ob=1,variance_tr=10000)
    KAP=Kalman_agent_parallels(environment,gamma=0.95,variance_ob=1,variance_tr=30,gamma_curiosity=.95, curiosity_factor=0.01,v_ob_curiosity=1,v_tr_curiosity=10)
    KAR=Kalman_agent_raffine(environment,gamma=0.95,variance_ob=50,variance_tr=10,gamma_curiosity=.9, curiosity_factor=0.1)
    KAS=Kalman_agent_softmax(environment,gamma=0.95,variance_ob=100,variance_tr=5,curiosity_factor=[(5000-10*i)/1000 for i in range(500)]+[0 for i in range(10000)])
    KARf=Kalman_agent_raffine_faux(environment,gamma=0.95,variance_ob=100,variance_tr=10,gamma_curiosity=0.4,curiosity_factor=3)
    KAPf=Kalman_agent_parallels_faux(environment,gamma=0.95,variance_ob=100,variance_tr=20,gamma_curiosity=0.4, curiosity_factor=1,v_ob_curiosity=50,v_tr_curiosity=20)
    reward_QA, counter_QA,value_table_QA=play(environment, QA, trials=1000, max_steps_per_episode=500, learn=True)
    reward_KA, counter_KA,table_mean_KA,table_variance_KA=play(environment, KA, trials=1000, max_steps_per_episode=500, learn=True)
    reward_KAP, counter_KAP, table_mean_KAP,table_variance_KAP,table_curiosity_KAP=play(environment, KAP, trials=1000, max_steps_per_episode=500, learn=True)
    reward_KAR, counter_KAR, table_mean_KAR, table_variance_KAR, table_curiosity_KAR=play(environment, KAR, trials=1000, max_steps_per_episode=500, learn=True)
    reward_KAS, counter_KAS, table_mean_KAS,table_variance_KAS=play(environment, KAS, trials=1000, max_steps_per_episode=500, learn=True)
    reward_KARf, counter_KARf, table_mean_KARf,table_variance_KARf,table_curiosity_KARf=play(environment, KARf, trials=1000, max_steps_per_episode=500, learn=True)
    reward_KAPf, counter_KAPf, table_mean_KAPf,table_variance_KAPf,table_curiosity_KAPf=play(environment, KAPf, trials=1000, max_steps_per_episode=500, learn=True)
    rewards[0].append(reward_QA[500:])
    rewards[1].append(reward_QA)
    rewards[2].append(reward_KA[500:])
    rewards[3].append(reward_KA)
    rewards[4].append(reward_KAP[500:])
    rewards[5].append(reward_KAP)
    rewards[6].append(reward_KAR[500:])
    rewards[7].append(reward_KAR)
    rewards[8].append(reward_KAS[500:])
    rewards[9].append(reward_KAS)
    rewards[10].append(reward_KARf[500:])
    rewards[11].append(reward_KARf)
    rewards[12].append(reward_KAPf[500:])
    rewards[13].append(reward_KAPf)
    
mean_rewards=[np.mean(reward) for reward in rewards]
print("avg_reward_QA 500+ = " +str(mean_rewards[0])+", "+"avg_reward_total_QA = "+str(mean_rewards[1]))
print("avg_reward_KA 500+ = " +str(mean_rewards[2])+", "+ "avg_reward_total_KA = "+str(mean_rewards[3]))
print("avg_reward_KAP 500+ = " +str(mean_rewards[4])+", "+"avg_reward_total_KAP = "+str(mean_rewards[5]))
print("avg_reward_KAR 500+ = " +str(mean_rewards[6])+", "+"avg_reward_total_KAR = "+str(mean_rewards[7]))
print("avg_reward_KAS 500+ = " +str(mean_rewards[8])+", "+"avg_reward_total_KAS = "+str(mean_rewards[9]))
print("avg_reward_KARf 500+ = " +str(mean_rewards[10])+", "+"avg_reward_total_KARf = "+str(mean_rewards[11]))
print("avg_reward_KAPf 500+ = " +str(mean_rewards[12])+", "+"avg_reward_total_KAPf = "+str(mean_rewards[13]))

#bestt,besttt=find_best_seven_Kalman_raffine()
#print(besttt,bestt[besttt])



plt.figure()
plt.plot(reward_QA, color='black',)
plt.xlabel("Numéro de l'essai (Q-learning)")



plt.figure()
plt.plot(reward_KA, color='black')
plt.xlabel("Essai KA")


plt.figure()
plt.plot(reward_KAP, color='black')
plt.xlabel("Essai KAP")

plt.figure()
plt.plot(reward_KAR, color='black')
plt.xlabel("Essai KAR")

plt.figure()
plt.plot(reward_KAS, color='black')
plt.xlabel("Essai KAS")

plt.figure()
plt.plot(reward_KARf, color='black')
plt.xlabel("Essai KARf")

plt.figure()
plt.plot(reward_KAPf, color='black')
plt.xlabel("Essai KAPf")


