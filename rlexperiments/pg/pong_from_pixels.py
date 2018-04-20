
'''
Andrej Karpathy's "Pong from Pixels" example. The code has been modified 
to use Tensorflow for optimization.

Blog post: 
  http://karpathy.github.io/2016/05/31/rl/

Original source:
    https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5

'''

import os
import numpy as np
import gym
import tensorflow as tf
from rlexperiments.common.tensorflow_reporter import TFReporter

class PolicyEstimator():
    '''Policy Function approximator'''
    
    def __init__(self, input_size, hidden_size, action_size, learning_rate=0.001):

        # Define the feed-forward part of the network

        self.state_input = tf.placeholder(shape=[None, input_size], dtype=tf.float32, name='state_input')
        
        self.fc1 = tf.contrib.layers.fully_connected(self.state_input, 
            hidden_size, 
            activation_fn=tf.nn.relu, 
            biases_initializer=None, 
            scope='fc1')

        self.fc2 = tf.contrib.layers.fully_connected(self.fc1, 
            hidden_size, 
            activation_fn=tf.nn.relu, 
            biases_initializer=None, 
            scope='fc2')

        self.logits = tf.contrib.layers.fully_connected(self.fc2, 
            action_size, 
            activation_fn=None,
            biases_initializer=None, 
            scope='logits')

        self.probs = tf.nn.softmax(self.logits)


        # Define the training procedure. 
        
        # We feed the reward and the sampled action into the network to compute 
        # the loss, and use it to update the network.
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)

        # 'responsible_outputs' are the probabilities associated with the actions 
        # that have been sampled
        self.actions = tf.one_hot(self.action_holder, action_size)
        self.responsible_outputs = tf.reduce_sum(self.probs * self.actions, axis=1)
        
        # The loss function increases the log probability for actions that worked 
        # and decrease it for actions that didn't.                
        self.advantage = self.reward_holder
        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.advantage)

        # calculate the gradients of the loss function w.r.t. all trainable variables
        tvars = tf.trainable_variables()
        self.gradients = tf.gradients(self.loss, tvars)
                                
        # define a gradient placeholder for each trainable variable
        self.gradient_holders = []
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)

        # add operations that update the weights
        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00025, decay=0.99, momentum=0.0, epsilon=1e-6)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))
    


    def predict(self, x, sess):
        feed_dict = { self.state_input: x }
        probs = sess.run(self.probs, feed_dict=feed_dict)
        return probs


    def compute_gradients(self, states, actions, rewards, sess):
        feed_dict = {
            self.state_input: states,     # begin state
            self.action_holder: actions,  # sampled action
            self.reward_holder: rewards   # discounted reward
        }
        p_loss, grads = sess.run([self.loss, self.gradients], feed_dict=feed_dict)
        return p_loss, grads


    def init_gradient_buffer(self, sess):
        # Create a buffer that will store gradients for all trainable parameters. 
        # Evaluating the list of trainable variables is an nice way to allocate
        # the correctly shaped data structures.
        self.grad_buffer = sess.run(tf.trainable_variables())
        self._reset_gradient_buffer()
    

    def collect_gradients(self, gradients):
        for idx, grad in enumerate(gradients):
            self.grad_buffer[idx] += grad
        
    def update(self, session):
        '''
        Update the network with the gradients that have been collected
        in the gradient buffer.
        '''
        feed_dict = dict(zip(self.gradient_holders, self.grad_buffer))
        session.run(self.update_batch, feed_dict=feed_dict)
        # reset the gradient buffer so we do not apply the gradients multiple times
        self._reset_gradient_buffer()

    def _reset_gradient_buffer(self):
        for ix, grad in enumerate(self.grad_buffer):
            self.grad_buffer[ix] = grad * 0



def discount_rewards(r, gamma):
  '''Take float array of rewards and compute discounted reward'''
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r


def preprocess(img):
  '''Preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector'''
  img = img[35:195]    # crop
  img = img[::2,::2,0] # downsample by factor of 2
  img[img == 144] = 0  # erase background (background type 1)
  img[img == 109] = 0  # erase background (background type 2)
  img[img != 0] = 1    # everything else (paddles, ball) just set to 1
  return img.astype(np.float).ravel()

def report_batch_grads(grads, episode_number, tf_reporter):
    print('Report batch gradients')
    _report_gradients(grads, 'BatchGradients', episode_number, tf_reporter)

def _report_gradients(grads, prefix, episode_number, tf_reporter):
    for i in range(len(grads)):
        var_name = tf.trainable_variables()[i].name
        tf_reporter.log_histogram(prefix+'/'+var_name, grads[i], episode_number)


def main():

    image_dim = 80 * 80
    learning_rate = 0.001
    hidden_size = 200
    action_size = 2
    gamma = 0.99
    batch_size = 10 # every how many episodes to do a param update?
    cuda_visible_devices = "1"
    gpu_memory_fraction = 0.3
    summary_path = '/tmp/summaries/pong-pg'
    model_path = '/tmp/models/pong-pg'
    render = False
    load_model = False


    if not os.path.exists(summary_path):
        os.makedirs(summary_path)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    tf.reset_default_graph()

    policy = PolicyEstimator(input_size=image_dim, hidden_size=hidden_size, action_size=action_size, learning_rate=learning_rate)

    saver = tf.train.Saver()

    env = gym.make("Pong-v0")
    observation = env.reset()

    prev_x = None         # used in computing the difference frame
    running_reward = None # moving average of episode rewards
    running_steps = None  # moving average of episode length
    steps = 0             # number of steps in the current episode
    total_steps = 0
    rounds = 0            # number of rounds in current episode
    reward_sum = 0
    episode_number = 0

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction

    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())

        if load_model == True:
            ckpt = tf.train.get_checkpoint_state(model_path)
            print('Loading model checkpoint: %s' % ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)


        summary_writer = tf.summary.FileWriter(summary_path)
        tf_reporter = TFReporter(summary_writer=summary_writer)

        policy.init_gradient_buffer(sess)

        ep_history = []

        while True:

            if render: env.render()

            # create a difference image as network input
            cur_x = preprocess(observation)
            x = cur_x - prev_x if prev_x is not None else np.zeros(image_dim)
            prev_x = cur_x

            # sample an action from the policy
            aprob = policy.predict(np.array([x]), sess)[0][0] # action probability: the probability of moving UP
            action = 0 if np.random.uniform() < aprob else 1
            
            # report the dense layer input
            if total_steps % 20 == 0 and total_steps != 0:
                tf_reporter.log_histogram('FCInput', x, total_steps)

            # execute the action
            pong_action = 2 if action == 0 else 3 # 2=UP, 3=DOWN
            observation, reward, done, info = env.step(pong_action)

            # append the step to the episode history
            ep_history.append([x, action, reward])
            reward_sum += reward
            steps += 1
            total_steps += 1

        
            if done: # an episode finished

                episode_number += 1

                ep_history = np.array(ep_history)

                # compute the discounted reward backwards through time
                discounted_epr = discount_rewards(ep_history[:, 2], gamma)

                # standardize the rewards to be unit normal (helps control the gradient estimator variance)
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)

                if episode_number % 20 == 0 and episode_number != 0:
                    tf_reporter.log_histogram('DiscountedRewards', discounted_epr, episode_number)


                # compute gradients
                p_loss, grads = policy.compute_gradients(np.vstack(ep_history[:,0]), ep_history[:,1], discounted_epr, sess)

                # collect the gradients until we update the network
                policy.collect_gradients(grads)

                # perform parameter update every batch_size episodes
                if episode_number % batch_size == 0 and episode_number != 0:
                    report_batch_grads(policy.grad_buffer, episode_number, tf_reporter)
                    print('Update policy')
                    policy.update(sess)
                
                # book-keeping
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                running_steps = steps if running_steps is None else running_steps * 0.99 + steps * 0.01
                print('episode %d: episode_reward=%f mean_reward=%f episode_length=%d mean_length=%d' % (episode_number, reward_sum, running_reward, steps, running_steps))

                observation = env.reset()
                reward_sum = 0
                prev_x = None
                steps = 0
                rounds = 0
                ep_history = []

                # Write summary statistics to tensorboard
                report_stats_freq = 20
                if episode_number % report_stats_freq == 0 and episode_number != 0:
                    summary = tf.Summary()
                    summary.value.add(tag='Info/Reward', simple_value=running_reward)
                    summary.value.add(tag='Info/Episode Length', simple_value=running_steps)
                    summary.value.add(tag='Info/Policy Loss', simple_value=float(p_loss))
                    summary_writer.add_summary(summary, episode_number)
                    summary_writer.flush()
                    print("Report summary")

                # Save agent's model
                save_model_freq = 200
                if episode_number % save_model_freq == 0 and episode_number != 0:
                    saver.save(sess, model_path+'/model-' + str(episode_number) + '.cptk')
                    print("Saved model")


            if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
                rounds += 1
                print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')


if __name__ == "__main__":
    main()