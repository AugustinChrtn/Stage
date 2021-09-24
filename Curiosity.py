import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from Gridworld import State
from Useful_functions import play, display, find_best_quatuor_Kalman_sum, find_best_triplet_Kalman,find_best_triplet_Q_learning,prefered_direction,heatmap_variance,find_best_seven_Kalman_parallels, find_best_five_Kalman_raffine
from Bigworld import BigState





from Q_learning import Q_Agent
from Kalman import Kalman_agent
from Kalman_parallels import Kalman_agent_parallels
from Kalman_raffine import Kalman_agent_raffine
from Kalman_sum import Kalman_agent_sum
from Kalman_delayed_sum import Kalman_agent_delayed_sum

environment=BigState()
rewards=[[]for i in range(12)]
for i in range(50):
        
        QA=Q_Agent(environment,alpha=0.95,gamma=1,beta=0.05)
        KA= Kalman_agent(environment,gamma=1,variance_ob=1,variance_tr=30000)
        KAP=Kalman_agent_parallels(environment,gamma=1,variance_ob=1,variance_tr=30000,gamma_curiosity=0.3, curiosity_factor=0.03,v_ob_curiosity=0.1,v_tr_curiosity=1)
        #KAR=Kalman_agent_raffine(environment,gamma=1,variance_ob=1,variance_tr=500,gamma_curiosity=1, curiosity_factor=0.01)
        KAS=Kalman_agent_sum(environment,gamma=1,variance_ob=1,variance_tr=30,curiosity_factor=2)
        KADS=Kalman_agent_delayed_sum(environment,gamma=1,variance_ob=1,variance_tr=40,curiosity_factor=2,gamma_curiosity=0.15,alpha=1)
        
        reward_QA,counter_QA,value_table_QA,result_every_photo_QA = play(environment, QA)
        reward_KA,counter_KA,table_mean_KA,table_variance_KA,result_every_photo_KA = play(environment, KA)
        reward_KAP,counter_KAP, table_mean_KAP,table_variance_KAP,table_curiosity_KAP, result_every_photo_KAP = play(environment, KAP)
        #reward_KAR,counter_KAR, table_mean_KAR, table_variance_KAR, table_curiosity_KAR, result_every_photo_KAR = play(environment, KAR)
        reward_KAS,counter_KAS, table_mean_KAS,table_variance_KAS, result_every_photo_KAS = play(environment, KAS)
        reward_KADS,counter_KADS,table_mean_KADS,table_variance_KADS,table_curiosity_KADS, result_every_photo_KADS = play(environment,KADS)
        print(i)
        
        rewards[0].append(reward_QA[500:])
        rewards[1].append(reward_QA)
        rewards[2].append(reward_KA[500:])
        rewards[3].append(reward_KA)
        rewards[4].append(reward_KAP[500:])
        rewards[5].append(reward_KAP)
        #rewards[6].append(reward_KAR[500:])
        #rewards[7].append(reward_KAR)
        rewards[8].append(reward_KAS[500:])
        rewards[9].append(reward_KAS)
        rewards[10].append(reward_KADS[500:])
        rewards[11].append(reward_KADS)
    
mean_rewards=[np.mean(reward) for reward in rewards]
print("avg_reward_QA 500+ = " +str(mean_rewards[0])+", "+"avg_reward_total_QA = "+str(mean_rewards[1]))
print("avg_reward_KA 500+ = " +str(mean_rewards[2])+", "+ "avg_reward_total_KA = "+str(mean_rewards[3]))
print("avg_reward_KAP 500+ = " +str(mean_rewards[4])+", "+"avg_reward_total_KAP = "+str(mean_rewards[5]))
#print("avg_reward_KAR 500+ = " +str(mean_rewards[6])+", "+"avg_reward_total_KAR = "+str(mean_rewards[7]))
print("avg_reward_KAS 500+ = " +str(mean_rewards[8])+", "+"avg_reward_total_KAS = "+str(mean_rewards[9]))
print("avg_reward_KADS 500+ = " +str(mean_rewards[10])+", "+"avg_reward_total_KADS = "+str(mean_rewards[11]))
print(" ")
        
"""plt.figure()
plt.plot(reward_QA, color='black')
plt.plot([64 for i in range(1000)],color='black',linestyle='--',label='Récompense optimale')
plt.xlabel("Numéro de l'essai (Q-learning softmax)")
plt.ylabel("Récompense")
plt.legend()

plt.figure()
plt.plot(reward_KA, color='black')
plt.plot([64 for i in range(1000)],color='black',linestyle='--',label='Récompense optimale')
plt.xlabel("Essai KA")
plt.ylabel("Récompense")
plt.legend()



plt.figure()
plt.plot(reward_KAP, color='black')
plt.plot([64 for i in range(1000)],color='black',linestyle='--',label='Récompense optimale')
plt.xlabel("Essai KAP")
plt.legend()

plt.figure()
plt.plot(reward_KAR, color='black')
plt.xlabel("Essai KAR") 

plt.figure()
plt.plot(reward_KAS, color='black')
plt.xlabel("Essai KAS") """


plt.figure()
plt.plot(reward_KADS, color='black')
plt.xlabel("Essai KADS")




photos=[0,10,50,100,200,300,500,800,1000]

"""
for i in range(len(result_every_photo_QA)):
    plt.figure()
    plt.title("Essai QA "+str(photos[i]))
    heatmap_variance(result_every_photo_QA[i],environment)



for i in range(len(result_every_photo_KA)):
    plt.figure()
    plt.title("Essai KA "+str(photos[i]))
    heatmap_variance(result_every_photo_KA[i],environment)

    
for i in range(len(result_every_photo_KAP)):
    plt.figure()
    plt.title("Essai KAP "+str(photos[i]))
    heatmap_variance(result_every_photo_KAP[i],environment) 
    

for i in range(len(result_every_photo_KAR)):
    plt.figure()
    plt.title("Essai KAR "+str(photos[i]))
    heatmap_variance(result_every_photo_KAR[i],environment)


for i in range(len(result_every_photo_KAS)):
    plt.figure()
    plt.title("Essai KAS "+str(photos[i]))
    heatmap_variance(result_every_photo_KAS[i],environment)

for i in range(len(result_every_photo_KADS)):
    plt.figure()
    plt.title("Essai KADS "+str(photos[i]))
    heatmap_variance(result_every_photo_KADS[i],environment)"""

#bestt,besttt=find_best_triplet_Kalman()
#print(besttt,bestt[besttt])




