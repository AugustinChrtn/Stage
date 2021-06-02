import numpy as np

class Kalman_raffine(): 
    
    def __init__(self, environment, gamma=0.8, variance_ob=10,variance_tr=0.1,gamma_curiosity=0.4,curiosity_factor=10,v_ob_curiosity=20,v_tr_curiosity=10):
        self.environment = environment
        self.KF_table_mean = dict()
        self.KF_table_variance = dict()
        self.KF_table_curiosity=dict()
        self.counter=dict()
        self.KF_table_cur_var=dict()
        for x in range(environment.height):
            for y in range(environment.width):
                self.KF_table_mean[(x,y)] = {'UP':0, 'DOWN':0, 'LEFT':0, 'RIGHT':0}
                self.KF_table_variance[(x,y)] = {'UP':1, 'DOWN':1, 'LEFT':1, 'RIGHT':1}
                self.KF_table_curiosity[(x,y)] = {'UP':1, 'DOWN':1, 'LEFT':1, 'RIGHT':1}
                self.KF_table_cur_var[(x,y)]={'UP':1, 'DOWN':1, 'LEFT':1, 'RIGHT':1}
                self.counter[(x,y)]={'UP':0, 'DOWN':0, 'LEFT':0, 'RIGHT':0}
        self.gamma = gamma
        self.gamma_curiosity=gamma_curiosity
        self.variance_ob=variance_ob
        self.variance_tr=variance_tr
        self.curiosity_factor=curiosity_factor
        self.v_ob_curiosity=v_ob_curiosity
        self.v_tr_curiosity=v_tr_curiosity
    
    def choose_action(self):
        get_mean=self.KF_table_mean[self.environment.current_location]
        get_variance=self.KF_table_variance[self.environment.current_location]
        get_curiosity=self.KF_table_curiosity[self.environment.current_location]
        get_cur_var=self.KF_table_cur_var[self.environment.current_location]
        dict_probas=dict()
        for move in ['UP','DOWN','LEFT','RIGHT']:
            dict_probas[move]=np.random.normal(get_mean[move],np.sqrt(get_variance[move]))+self.curiosity_factor*np.random.normal(get_curiosity[move],np.sqrt(get_cur_var[move]))
        dic_values=list(dict_probas.values())
        action = list(dict_probas.keys())[dic_values.index(max(dic_values))]
        self.counter[self.environment.current_location][action]+=1
        return action
    

    def learn(self, old_state, reward, new_state, action):
        """Updates the mean and the variance using two parallel Kalman algorithms"""
        means_new_state = self.KF_table_mean[new_state]
        max_mean_in_new_state = max(means_new_state.values())
        current_mean = self.KF_table_mean[old_state][action]
        curiosity_new_state=self.KF_table_curiosity[new_state]
        max_curiosity_in_new_state=max(curiosity_new_state.values())
        current_variance = self.KF_table_variance[old_state][action]
        current_cur_var=self.KF_table_cur_var[old_state][action]
        self.KF_table_mean[old_state][action] = ((current_variance+self.variance_tr)*(reward +self.gamma * max_mean_in_new_state)+(self.variance_ob*current_mean))/ (current_variance+self.variance_tr+self.variance_ob)
        self.KF_table_variance[old_state][action]=((current_variance+self.variance_tr)*self.variance_ob)/(current_variance+self.variance_ob+self.variance_tr)
        self.KF_table_curiosity[old_state][action] = ((current_cur_var+self.v_tr_curiosity)*(self.KF_table_variance[old_state][action]+self.gamma_curiosity*max_curiosity_in_new_state)+(self.v_ob_curiosity*current_cur_var))/ (current_cur_var+self.v_tr_curiosity+self.v_ob_curiosity)
        self.KF_table_cur_var[old_state][action]=((current_cur_var+self.v_tr_curiosity)*self.v_ob_curiosity)/(current_cur_var+self.v_ob_curiosity+self.v_tr_curiosity)
        for move in ['UP','DOWN','LEFT','RIGHT']:
            if move != action : 
                self.KF_table_variance[old_state][move]+=self.variance_tr
                self.KF_table_cur_var[old_state][move]+=self.v_tr_curiosity