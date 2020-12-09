import tensorflow as tf
import numpy as np
import os, time, copy, math
from bridge_env import BridgeEnv
from core import DeepQNetwork
from utils import processsa
import matplotlib.pyplot as plt
from model_generate import Predictor

episodes = 50
env = BridgeEnv()

update_period = 5

agent = DeepQNetwork(
    env.n_actions, env.n_features,
    learning_rate = 0.003, 
    reward_decay = 0.9,
    e_greedy = 0.99,
    e_greedy_increment = None,
    use = 'application'
)

costs = []
actions = np.zeros([episodes, 50], dtype=np.int32)
states = np.zeros([episodes, 50], dtype=np.int32)
#record the failure number in each episode
failure = np.zeros([episodes, 1], dtype=np.int32)

#start safe DQN
step = 0
for episode in range(episodes):
    #each episode in MC simulation starts with initial network parameters(stored by checkpoint)
    #to imitate each new situation
    print("episode:"+str(episode))
    agent.reset()
    observation = copy.deepcopy(env.reset())
    # reset the predictor
    env.pred.reset()
    # reset the transition matrix in env
    env.trans_his = env.pred.transition
    #print(env.trans_his)
    env.his_pred_accuracy = env.pred.accuracy

    done = False
    t, year, rAll, step = 0,0,0,0
    while not done:
        # for every update_period, update the transition model
        a, method = agent.choose_action(observation, way='safe')
        if method=='exploitation': 
            # execute action directly
            states[episode, t] = copy.deepcopy(env.state_num)
            actions[episode, t] = copy.deepcopy(a)
            observation_, reward, fail, done = env.step(a, real_world=True)
            cost = env.spending(a, fail)
            age = env.time
            env.pred.store_transition(observation, a, observation_, age)
            
            agent.safeQlearn(observation, a, reward, observation_, step)

            observation = copy.deepcopy(observation_)
            t += 1
            year += 2
            rAll += cost*math.pow(agent.gamma, year)

        else:
            #predict the next state according to the model
            #store the virtual transition
            #then choose the optimal action without being stored

            #a is a random action, predict the result
            observation_, reward, fail, done = env.step(a, real_world=False, noChange=True)
            age = env.time+2
            env.pred.store_transition(observation, a, observation_, age)
            agent.safeQlearn(observation, a, reward, observation_, step)
            
            a_opt = agent.choose_action(observation, way='greedy')
            states[episode, t] = copy.deepcopy(env.state_num)
            actions[episode, t] = copy.deepcopy(a_opt)
            #execute the optimal action, real world=True
            observation_, reward, fail, done = env.step(a_opt, real_world=True)
            cost = env.spending(a_opt, fail)

            observation = copy.deepcopy(observation_)
            t += 1
            year += 2
            rAll += cost*math.pow(agent.gamma, year)
        
        #record failure
        if fail:
            failure[episode] += 1

        if t>0 and t%update_period==0:
            #update transition model
            env.pred.update()
            env.trans_his = env.pred.transition

        step += 1

        if episode%20 == 0:
            # 需要修改，只记录第100年
            env.pred.save_transition(episode=episode, year=year-2)

    costs.append(rAll)


print(failure.mean())
path= 'result\\simulation\\onlineDQN'
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

print(states[2])
print(actions[2])

print(states[4])
print(actions[4])

print(states[6])
print(actions[6])

print(states[8])
print(actions[8])

'''
print(states[20])
print(actions[20])

print(states[40])
print(actions[40])

print(states[60])
print(actions[60])

print(states[80])
print(actions[80])
'''

#plot costs
plt.plot(np.arange(len(costs)), costs)
plt.ylabel('Cost')
plt.xlabel('simulation')
plt.show()