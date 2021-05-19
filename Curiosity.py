import numpy as np
import matplotlib.pyplot as plt
from Gridworld import State
from Kalman_learning_agent import Kalman_agent
from Kalman_learning_agent_parallels import Kalman_agent_parallels
from Q_learning_agent import Q_Agent
from Useful_functions import play, display, find_best_triplet_Kalman,find_best_triplet_Q_learning,chosen_direction,gradient_variance_max,find_best_quatuor_Kalman_parallels

environment = State()
#Each agent is set with optimized values
KA = Kalman_agent(environment,gamma=0.8,variance_ob=1,variance_tr=0.01)
QA=Q_Agent(environment,epsilon=0.01,alpha=0.26,gamma=0.7)
KAP=Kalman_agent_parallels(environment,gamma=0.8,variance_ob=1,variance_tr=0.01,gamma_curiosity=0.3)

reward_QA, counter_QA, value_table_QA=play(environment, QA, trials=1000, max_steps_per_episode=1000, learn=True)
reward_KA,counter_KA,table_mean_KA,table_variance_KA=play(environment, KA, trials=1000, max_steps_per_episode=1000, learn=True)
reward_KAP, counter_KAP, table_mean_KAP,table_variance_KAP,table_curiosity_KAP=play(environment, KAP, trials=1000, max_steps_per_episode=1000, learn=True)

"""rewards=[[]for i in range(3)]
for i in range(10):
    KA= Kalman_agent(environment,gamma=0.8,variance_ob=1,variance_tr=0.01)
    QA=Q_Agent(environment,epsilon=0.01,alpha=0.26,gamma=0.7)
    KAP=Kalman_agent_parallels(environment,gamma=0.8,variance_ob=1,variance_tr=0.01,gamma_curiosity=0.3)
    reward_QA,value_table_QA=play(environment, QA, trials=1000, max_steps_per_episode=1000, learn=True)
    reward_KA,table_mean_KA,table_variance_KA=play(environment, KA, trials=1000, max_steps_per_episode=1000, learn=True)
    reward_KAP,table_mean_KAP,table_variance_KAP,table_curiosity_KAP=play(environment, KAP, trials=1000, max_steps_per_episode=1000, learn=True)
    rewards[0].append(reward_QA[200:])
    rewards[1].append(reward_KA[200:])
    rewards[2].append(reward_KAP[200:])
mean_rewards=[np.mean(reward) for reward in rewards]
print("avg_reward_QA = " +str(mean_rewards[0]))
print("avg_reward_KA = " +str(mean_rewards[1]))
print("avg_reward_KAP = " +str(mean_rewards[2]))"""


plt.figure()
plt.plot(reward_QA, color='black')
plt.xlabel("Essai QA")
plt.ylabel("Récompense")

plt.figure()
plt.plot(reward_KA, color='black')
plt.xlabel("Essai KA")
plt.ylabel("Récompense")

plt.figure()
plt.plot(reward_KAP, color='black')
plt.xlabel("Essai KAP")
plt.ylabel("Récompense")

