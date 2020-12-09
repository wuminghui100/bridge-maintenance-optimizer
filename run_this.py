from bridge_env import BridgeEnv
from core import DeepQNetwork
import os, math, time, copy
import numpy as np
import tensorflow as tf

def run_env():
    start = time.time()
    step = 0
    for episode in range(15001):
        year = 0
        if episode%10 == 0:
            env.lifereward_his.append(0)
        observation = copy.deepcopy(env.reset())
        
        #print('episode:'+str(episode))
        while True:
            action = RL.choose_action(observation)
            observation_, reward, _, done = env.step(action)
            RL.store_transition(observation, action, reward, observation_)
            if episode%10 == 0:
                env.lifereward_his[int(episode/10)] += reward*math.pow(RL.gamma, year)

            if step > 2000 and step%5 == 0:
                RL.learn(episode, step)
            
            observation = copy.deepcopy(observation_)
            
            if done:
                break
            step += 1
            #modify
            year += 2
        
        if episode%1000 == 0 and episode>0:
            filepath = 'result\\training results\\train14\\episode' + str(episode)
            dirpath = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(dirpath, filepath)
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            #saver.save(RL.sess, filepath+'\\trainning-'+str(episode)+'.cpkt')
            #saver.save(sess, filepath+'/traning-'+str(episode)+'.cpkt')
            np.save(filepath+'\\reward.npy', env.lifereward_his)
            np.save(filepath+'\\DQNloss.npy', RL.DQNloss)
            #np.save(filepath+'\\Q_act.npt', RL.q_act)
            print('Save model')
            elapsed = time.time()-start
            print(episode, elapsed)
            start = time.time()


if __name__ == "__main__":
    env = BridgeEnv()
    RL = DeepQNetwork(
        env.n_actions, env.n_features,
        learning_rate = 0.00025,
        learning_rate_increment = 0.00005,
        reward_decay = 0.9,
        e_greedy = 0.99,
        e_greedy_increment = 0.00005, 
    )
    run_env()

    RL.plot_cost()
    env.plot_lifecost()
    