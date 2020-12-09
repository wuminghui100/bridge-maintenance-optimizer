import numpy as np
import pandas as pd
import tensorflow as tf
import os

np.random.seed(1)
tf.set_random_seed(1)

#Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate=0.005,
        reward_decay=0.9,
        e_greedy=0.99,
        epsilon_app=0.8,
        replace_target_iter=300,
        memory_size=500000,
        batch_size=10000,
        e_greedy_increment = None,
        learning_rate_increment = None,
        output_graph=False,
        use = 'train'
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.lr_min = learning_rate
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_app = epsilon_app
        self.epsilon_increment = e_greedy_increment
        self.epsilon =0 if e_greedy_increment is not None else self.epsilon_max
        self.lr_increment = learning_rate_increment
        self.lr = 0.01 if learning_rate_increment is not None else self.lr_min
        self.use = use

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()

        self.sess = tf.Session()
        t_params = tf.get_collection('target_net_params')
        a_params = tf.get_collection('act_net_params')
        #self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, a_params)]

        #self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        # import
        if self.use == 'application':
            self._import_variable()

        self.loss_his = []

    def _build_net(self):
        # ---action net---
        self.s = tf.placeholder(tf.int32, [None, self.n_features], name='s')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')
        with tf.variable_scope('act_net'):
            c_names, n_l1, w_initializer, b_initializer=\
                ['act_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 24, \
                tf.random_normal_initializer(0.,0.3), tf.constant_initializer(0.1)
            
            # 1st layer
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(tf.to_float(self.s), w1)+b1)
            
            # 2rd layer
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_act = tf.matmul(l1, w2)+b2
            
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_act))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
        
        # ---target net---
        self.s_ = tf.placeholder(tf.int32, [None, self.n_features], name='s_')
        with tf.variable_scope('targt_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
        
            # 1st layer
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1',[self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1',[1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(tf.to_float(self.s_), w1)+b1)
            
            # 2rd layer
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2)+b2
        # saver
        self.saver = tf.train.Saver(max_to_keep=5)

    def _import_variable(self):
        path = 'result\\training results\\episode20000'
        dirpath = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(dirpath, path)
        moudke_file = tf.train.latest_checkpoint(path)
        self.saver.restore(self.sess, moudke_file)

    def reset(self):
        #reset to the initial parameters(trained by historical data, without adjustment)
        self._import_variable()
  
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a,r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index,:] = transition
        self.memory_counter += 1
    
    def choose_action(self, observation, way='e-greedy'):
        observation = observation[np.newaxis, :]
        
        # if following totally greedy policy(in application)
        if way=='greedy':
            actions_value = self.sess.run(self.q_act, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
            return action
        # if following safe exploration policy (in application)
        if way=='safe':
            if np.random.uniform() < self.epsilon_app:
                actions_value = self.sess.run(self.q_act, feed_dict={self.s: observation})
                action = np.argmax(actions_value)
                return action,'exploitation'
            else:
                action = np.random.randint(0, self.n_actions)
                return action,'exploration'
        # if following e-greedy policy
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_act, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action
    
    def _replacement_target_params(self):
        t_params = tf.get_collection('target_net_params')
        a_params = tf.get_collection('act_net_params')
        self.sess.run([tf.assign(t, a) for t, a in zip(t_params, a_params)])
    
    
    def safeQlearn(self, s, a, r, s_, step):
        # bootstrapping
        self._replacement_target_params()

        s = s[np.newaxis, :]
        s_ = s_[np.newaxis, :]
        q_next, q_act = self.sess.run(
            [self.q_next, self.q_act],
            feed_dict = {
                self.s_: s_,
                self.s: s,
            })
        q_target = q_act.copy()
        q_target[0,a] = r + self.gamma*np.max(q_next, axis=1)
        _, self.DQNloss = self.sess.run([self._train_op, self.loss],
                                    feed_dict={self.s: s,
                                               self.q_target:q_target})
        self.loss_his.append(self.DQNloss)
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon<self.epsilon_max else self.epsilon_max
        self.lr = self.lr - self.lr_increment if self.lr>self.lr_min else self.lr_min
        self.learn_step_counter += 1

        if step>0 and step%10==0:
            filepath = 'result\\simulation\\onlineDQN\\'
            dirpath = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(dirpath, filepath)
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            saver = tf.train.Saver()
            saver.save(self.sess, filepath+'step-'+str(step))


    def learn(self, episode, step):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replacement_target_params()
            #print('\ntarget_params_replace')
        
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_act = self.sess.run(
            [self.q_next, self.q_act],
            feed_dict = {
                self.s_: batch_memory[:, -self.n_features:],
                self.s: batch_memory[:, :self.n_features],
            })
            
        q_target = q_act.copy()
            
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        _, self.DQNloss = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.loss_his.append(self.DQNloss)
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon<self.epsilon_max else self.epsilon_max
        self.lr = self.lr - self.lr_increment if self.lr>self.lr_min else self.lr_min
        self.learn_step_counter += 1

        if episode % 1000 == 0 and episode>0:
            filepath = 'result\\training results\\episode' + str(episode)
            dirpath = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(dirpath, filepath)
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            checkpoint_path = os.path.join(filepath+'\\training-'+str(episode)+'.ckpt')
            self.saver.save(self.sess, checkpoint_path, global_step=step)

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.loss_his)), self.loss_his)
        plt.ylabel('Loss')
        plt.xlabel('training steps')
        plt.show()