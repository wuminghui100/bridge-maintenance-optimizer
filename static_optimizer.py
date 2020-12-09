import tensorflow as tf
import numpy as np
import os, time, copy, math
from bridge_env import BridgeEnv
from core import DeepQNetwork
from utils import processsa
import matplotlib.pyplot as plt
import math

episodes = 1000
env = BridgeEnv()

agent = DeepQNetwork(
    env.n_actions, env.n_features,
    learning_rate = 0.0005, 
    reward_decay = 0.9,
    e_greedy = 0.99,
    e_greedy_increment = 0.005,
    use = 'application'
)

costs = []
actions = np.zeros([episodes, 50], dtype=np.int32)
states = np.zeros([episodes, 50], dtype=np.int32)
failure = np.zeros([episodes, 1], dtype=np.int32)
lifecost = []
for episode in range(episodes):
    observation = copy.deepcopy(env.reset())
    done = False
    t, year, rAll = 0,0,0
    while not done:
        a = agent.choose_action(observation, way='greedy')
        states[episode, t] = copy.deepcopy(env.state_num)
        actions[episode, t] = copy.deepcopy(a)
        observation_, _, fail, done = env.step(a, real_world=True)
        #real world cost
        cost = env.spending(a, fail)
        # temporarily ignore reward decay
        t += 1
        year += 2
        rAll += cost*math.pow(agent.gamma, year)
        observation = copy.deepcopy(observation_)
        if episode == 0:
            lifecost.append(cost)
        #record failure
        if fail:
            failure[episode] += 1

    #print(rAll)
    costs.append(rAll)

print(failure.mean())
path = 'result\\simulation\\staticDQN'
dirpath = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(dirpath, path)
if not os.path.exists(path):
    os.makedirs(path)
np.save(path+'\\cost.npy', costs)
np.save(path+'\\states.npy', states)
np.save(path+'\\actions.npy', actions)

#print average lifecycle cost
print(np.mean(costs))

print(states[1])
print(actions[1])

print(states[20])
print(actions[20])

print(states[40])
print(actions[40])

print(states[60])
print(actions[60])

print(states[80])
print(actions[80])

#plot lifereward
plt.plot(np.arange(len(lifecost)), lifecost)
plt.ylabel('Lifecost')
plt.xlabel('years')
plt.show()

#plot costs
plt.plot(np.arange(len(costs)), costs)
plt.ylabel('Cost')
plt.xlabel('simulation')
plt.show()