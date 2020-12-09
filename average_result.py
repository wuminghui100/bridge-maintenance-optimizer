import numpy as np
import os
import matplotlib.pyplot as plt

def plot_reward(reward):
    plt.plot(np.arange(len(reward)), reward)
    plt.ylabel('Reward')
    plt.xlabel('Training steps')
    plt.show()

reward = np.zeros(shape=[13, 1501])
average = np.zeros(shape=[1, 1501])
for train in range(1, 13):
    filepath = 'result\\training results\\train'+str(train)+'\\episode15000\\reward.npy'
    dirpath = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(dirpath, filepath)
    reward[train-1] = np.load(filepath)

#std
print(np.std(reward, axis=0))
# calculate average
average = np.mean(reward, axis=0)
plot_reward(average)