
import numpy as np
import tensorflow as tf

class Runner(object):

    def __init__(self, sess, env, replay_buffer, num_actions, nsteps, exploration, mainQN, valid_actions):
        self.sess = sess
        self.env = env
        self.replay_buffer = replay_buffer
        self.num_actions = num_actions
        self.nsteps = nsteps
        self.exploration = exploration
        self.mainQN = mainQN 
        self.valid_actions = valid_actions

        self.state = env.reset()
        self.episode_reward = 0
        self.episode_length = 0
        self.mean_ep_reward = None
        self.mean_ep_length = None
        self.total_steps = 0


    def run(self):

        for _ in range(self.nsteps):

            # do random exploration with prob. epsilon, otherwise sample
            # an action from the main-QN
            epsilon = self.exploration.value(self.total_steps)
            if np.random.rand(1) < epsilon:
                action = np.random.randint(0, self.num_actions)
            else:
                feed_dict = { self.mainQN.state_input: [np.array(self.state)] }
                action = self.sess.run(self.mainQN.max_q_action, feed_dict=feed_dict)[0]

            next_state, reward, done, _ = self.env.step(self.valid_actions[action])
            self.replay_buffer.add(self.state, action, reward, next_state, float(done))

            self.episode_length += 1
            self.episode_reward += reward
            self.total_steps += 1
        
            if done:
                self.mean_ep_reward = self.episode_reward if self.mean_ep_reward is None else self.mean_ep_reward * 0.99 + self.episode_reward * 0.01
                self.mean_ep_length = self.episode_length if self.mean_ep_length is None else self.mean_ep_length * 0.99 + self.episode_length * 0.01
                self.episode_reward = 0
                self.episode_length = 0
                self.state = self.env.reset()
            else:
                self.state = next_state
