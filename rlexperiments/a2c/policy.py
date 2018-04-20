
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from rlexperiments.common.util import ortho_init
from rlexperiments.common.distributions import CategoricalDist

class Policy(object):

    def __init__(self, sess, num_actions, num_env, num_steps, scope='model', reuse=False):

        num_batch = num_env * num_steps
        
        X = tf.placeholder(shape=[num_batch, 84, 84, 4], dtype=tf.int8, name="state_input")

        with tf.variable_scope(scope, reuse=reuse):

            conv1 = layers.convolution2d(tf.cast(X, tf.float32) / 255.0,
                            num_outputs=32,
                            kernel_size=8,
                            stride=4,
                            activation_fn=tf.nn.relu,
                            weights_initializer=ortho_init(np.sqrt(2)),
                            scope='conv1')
    
            conv2 = layers.convolution2d(conv1,
                            num_outputs=64,
                            kernel_size=4,
                            stride=2,
                            activation_fn=tf.nn.relu,
                            weights_initializer=ortho_init(np.sqrt(2)),
                            scope='conv2')

            conv3 = layers.convolution2d(conv2,
                            num_outputs=64,
                            kernel_size=3,
                            stride=1,
                            activation_fn=tf.nn.relu,
                            weights_initializer=ortho_init(np.sqrt(2)),
                            scope='conv3')

            flattened = layers.flatten(conv3)

            fc1 = tf.contrib.layers.fully_connected(flattened, 
                            512, 
                            activation_fn=tf.nn.relu, 
                            weights_initializer=ortho_init(np.sqrt(2)),
                            scope='fc1')

            # unnormalized logits, shape: (batch_size, num_actions)
            pi = tf.contrib.layers.fully_connected(fc1, 
                    num_actions, 
                    activation_fn=None, 
                    biases_initializer=None,
                    scope='pi')

            # state-value function, shape: (batch_size:, 1)
            vf = tf.contrib.layers.fully_connected(fc1, 
                    1, 
                    activation_fn=None, 
                    biases_initializer=None,
                    scope='vf')

        self.pd = CategoricalDist(pi)
        v0 = vf[:, 0] # reshape (20, 1) to (20,)
        a0 = self.pd.sample() # index of the most likely logit (with some noise)
        
        def step(ob):
            '''
              Input: 
                obs, shape: (num_env * num_steps, 84, 84, 4)
              Returns:
                action array, shape: (num_env * num_steps,) 
                value array, shape: (num_env * num_steps,)
            '''
            a, v = sess.run([a0, v0], {X: ob})
            return a, v

        def value(ob):
            '''
              Input: 
                obs, shape: (num_env * num_steps, 84, 84, 4)
              Returns:
                value array, shape: (num_env * num_steps,)
            '''            
            return sess.run(v0, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
