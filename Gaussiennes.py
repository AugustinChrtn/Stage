import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi, exp

domaine = range(-60,70)
mu_1,mu_2,mu_3,mu_4 = 1,-2,0,0.5
sigma_1,sigma_2,sigma_3,sigma_4 = 0.75,1,1.5,2
def f(mu,sigma,x):
    return 1/(sqrt(2*pi*pow(sigma,2))) * exp(-pow((x-mu),2)/(2*pow(sigma,2)))

y_1= [f(mu_1,sigma_1, x/10) for x in domaine]
y_2 = [f(mu_2,sigma_2, x/10) for x in domaine]
y_3 = [f(mu_3,sigma_3, x/10) for x in domaine]
y_4 = [f(mu_4,sigma_4, x/10) for x in domaine]
x_1=np.random.normal(mu_1,sigma_1)
x_2=np.random.normal(mu_2,sigma_2)
x_3=np.random.normal(mu_3,sigma_3)
x_4=np.random.normal(mu_4,sigma_4)

plt.figure(figsize=(6,4))
plt.plot(np.array(domaine)/10, y_1,label='µ='+str(mu_1)+', σ='+str(sigma_1),color='blue')
plt.plot(np.array(domaine)/10, y_2,label='µ='+str(mu_2)+', σ='+str(sigma_2),color='black')
plt.plot(np.array(domaine)/10, y_3,label='µ='+str(mu_3)+', σ='+str(sigma_3),color='green')
plt.plot(np.array(domaine)/10, y_4,label='µ='+str(mu_4)+', σ='+str(sigma_4),color='red')
plt.scatter([x_1,x_2,x_3,x_4],[f(mu_1,sigma_1, x_1),f(mu_2,sigma_2, x_2),f(mu_3,sigma_3, x_3),f(mu_4,sigma_4, x_4)], color=['blue','black','green','red'])
plt.annotate('Haut',color='black',xy=(-2.50,0.41))
plt.annotate('Droite',color='green',xy=(-2.50,0.2))
plt.annotate('Bas',color='red',xy=(3.6,0.1))
plt.annotate('Gauche',color='blue',xy=(-1.2,0.5))
plt.legend()
plt.ylabel('Probabilité')
