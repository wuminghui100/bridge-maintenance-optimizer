import numpy as np
from utils import time_encoder, processsa
import math,os
from model_generate import Predictor

class BridgeEnv:
    def __init__(self):
        self.time = 0
        self.n_features = 12
        self.action_space = [0,1,2,3]
        self.n_actions = len(self.action_space)
        self.state = self.reset()
        self.state_num = np.where(self.state[0:5]==1)
        self.lifereward_his = []
        self.trans_his = np.zeros(shape=[10,4,5,5])
        self.trans_real = np.zeros(shape=[10,4,5,5])
        self.pred = Predictor(1, co_phy=2) 
        #import historical transition from Predicor
        self.trans_his = self.pred.transition
        self.his_pred_accuracy = self.pred.accuracy
        #import real transition model from scenario files
        path = 'scenarios\\scenario6.npy'
        dirpath = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(dirpath, path)
        self.trans_real = np.load(path)
  

    def reset(self):
        state = np.zeros(self.n_features, dtype=np.int32)
        state[4] = 1
        self.time = 0
        _year = np.zeros(7, dtype = np.int32)
        _year = time_encoder(self.time)
        state[5:12] = _year
        self.state = state
        self.state_num = 4

        return self.state

    def render(self):
        print(self.state)

    def cost_normalize(self, cost):
        return (6.57e-7)*cost-0.2
    
    def cost_denormalize(self, cost_n):
        return (cost_n+0.2)/(6.57e-7)

    def costs(self, state, action, fail):
        a_costlist = [0, 70434, 251606, 732799]
        s_failpossibility = [0.1, 0.06, 0.01, 0.003, 0.0001]
        failcost = 1976947
        '''
        if fail:
            cost = a_costlist[action]+s_failpossibility[state]*failcost+failcost
        else:
            cost = a_costlist[action]+s_failpossibility[state]*failcost
        '''
        if fail:
            cost = a_costlist[action]+failcost
        else:
            cost = a_costlist[action]

        return self.cost_normalize(cost)
    
    def spending(self, action, fail):
        #real expenditure
        #assume theory suits well with real world case 
        a_costlist = [0, 70434, 251606, 732799]
        failcost = 1976947
        if fail:
            cost = a_costlist[action]+failcost
        else:
            cost = a_costlist[action]
        return cost

    def step(self, action, render=False, real_world=False, noChange=False):
        if noChange == False:
            self.time += 2
        #print(self.time)
        if self.time>=100:
            done = True
        else:
            done = False
        #transition model
        trans = np.zeros(shape=[4,5,5])
        #theoritical result based on data from other bridges
        if real_world==False:
            period = math.floor((self.time-1)/10.0)
            trans = self.trans_his[period]
        #real world transition based on reasonable assumption
        #still need more subtle design
        else:
            period = math.floor((self.time-1)/10.0)
            trans = self.trans_real[period]
        # deterioration
        sindex = self.state_num
        _sindex = np.random.choice(5, 1, p=trans[action, sindex, :])
        # failure
        s_failpossibility = [0.1, 0.06, 0.01, 0.003, 0.0001]
        if np.random.rand()<s_failpossibility[int(_sindex)]:
            # the deck system break down
            fail = True
        else:
            fail = False

        #reward, decided by the action and the possible failure
        reward = -self.costs(int(_sindex), action, fail)

        # if fail, the rebuild will lead the state back to state 4
        if fail:
            _state = 4
        else:
            _state = int(_sindex)

        if render:
            self.render()

        if noChange == False:
            # not prediction in safeQLearning
            # the state will change
            self.state_num = _state
            self.state[0:5] = processsa(self.state_num)
            self.state[5:12] = time_encoder(self.time)
            return self.state, reward, fail, done
        else:
            # state won't change
            # return the predicted next state
            state_num_vir = _state
            state_vir = np.zeros(self.n_features, dtype=np.int32)
            state_vir[0:5] = processsa(state_num_vir)
            state_vir[5:12] = time_encoder(self.time+2)
            return state_vir, reward, fail, done

    def plot_lifecost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.lifereward_his)), self.lifereward_his)
        plt.ylabel('Lifereward')
        plt.xlabel('episode/100')
        plt.show()