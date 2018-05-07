import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from rlexperiments.common.util import ortho_init
from rlexperiments.common.distributions import CategoricalPd

class CnnPolicy(object):

    def __init__(self, name, sess, ob_space, ac_space, nbatch): 
        with tf.variable_scope(name):
            self._init(sess, ob_space, ac_space, nbatch)
            self.scope = tf.get_variable_scope().name

    def _init(self, sess, ob_space, ac_space, nbatch): 
        nact = ac_space.n
        num_batch = None
        X = tf.placeholder(shape=[num_batch, 84, 84, 4], dtype=tf.int8, name="X")

        X_scaled = tf.cast(X, tf.float32) / 255.0

        with tf.variable_scope("pol"):

            pol_conv1 = layers.convolution2d(X_scaled,
                                num_outputs=32,
                                kernel_size=8,
                                stride=4,
                                activation_fn=tf.nn.relu,
                                weights_initializer=ortho_init(np.sqrt(2)),
                                scope='pol_conv1')
        
            pol_conv2 = layers.convolution2d(pol_conv1,
                            num_outputs=64,
                            kernel_size=4,
                            stride=2,
                            activation_fn=tf.nn.relu,
                            weights_initializer=ortho_init(np.sqrt(2)),
                            scope='pol_conv2')

            pol_conv3 = layers.convolution2d(pol_conv2,
                            num_outputs=64,
                            kernel_size=3,
                            stride=1,
                            activation_fn=tf.nn.relu,
                            weights_initializer=ortho_init(np.sqrt(2)),
                            scope='pol_conv3')

            pol_flattened = layers.flatten(pol_conv3)

            pol_fc = tf.contrib.layers.fully_connected(pol_flattened, 
                            512, 
                            activation_fn=tf.nn.relu, 
                            weights_initializer=ortho_init(np.sqrt(2)),
                            scope='pol_fc')

            pi = tf.contrib.layers.fully_connected(pol_fc, 
                    nact, 
                    activation_fn=None, 
                    weights_initializer=ortho_init(0.01),
                    scope='pi')

            self.pd = CategoricalPd(pi)


        with tf.variable_scope("vf"):

            vf_conv1 = layers.convolution2d(X_scaled,
                                num_outputs=32,
                                kernel_size=8,
                                stride=4,
                                activation_fn=tf.nn.relu,
                                weights_initializer=ortho_init(np.sqrt(2)),
                                scope='vf_conv1')
        
            vf_conv2 = layers.convolution2d(vf_conv1,
                            num_outputs=64,
                            kernel_size=4,
                            stride=2,
                            activation_fn=tf.nn.relu,
                            weights_initializer=ortho_init(np.sqrt(2)),
                            scope='vf_conv2')

            vf_conv3 = layers.convolution2d(vf_conv2,
                            num_outputs=64,
                            kernel_size=3,
                            stride=1,
                            activation_fn=tf.nn.relu,
                            weights_initializer=ortho_init(np.sqrt(2)),
                            scope='vf_conv3')

            vf_flattened = layers.flatten(vf_conv3)

            vf_fc = tf.contrib.layers.fully_connected(vf_flattened, 
                            512, 
                            activation_fn=tf.nn.relu, 
                            weights_initializer=ortho_init(np.sqrt(2)),
                            scope='vf_fc')

            self.vpred = tf.contrib.layers.fully_connected(vf_fc, 
                    1, 
                    activation_fn=None, 
                    weights_initializer=ortho_init(0.01),
                    scope='value')[:,0]

        ac = self.pd.sample()

        def step(ob):
            a, v = sess.run([ac, self.vpred], {X:ob})
            return a, v

        self.step = step    


    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
