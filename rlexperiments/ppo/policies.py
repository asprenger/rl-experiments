import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from rlexperiments.common.util import ortho_init
from rlexperiments.common.distributions import DiagGaussianPd, CategoricalPd


class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): 
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, 4)
        nact = ac_space.n

        sample_dtype = tf.int32
        sample_shape = []
        def sample_placeholder(prepend_shape, name=None):
            return tf.placeholder(dtype=sample_dtype, shape=prepend_shape+sample_shape, name=name)

        X = tf.placeholder(tf.uint8, ob_shape, name='X') 
        with tf.variable_scope("model", reuse=reuse):
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

            pi = tf.contrib.layers.fully_connected(fc1, 
                    nact, 
                    activation_fn=None, 
                    weights_initializer=ortho_init(0.01),
                    scope='pi')

            vf = tf.contrib.layers.fully_connected(fc1, 
                    1, 
                    activation_fn=None, 
                    weights_initializer=ortho_init(1.0),
                    scope='vf')[:,0]

        self.pd = CategoricalPd(pi)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)

        def step(ob):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, neglogp

        def value(ob):
            return sess.run(vf, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        self.sample_placeholder = sample_placeholder



class MlpPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): 
        ob_shape = (nbatch,) + ob_space.shape
        actdim = ac_space.shape[0]

        sample_dtype = tf.float32
        sample_shape = [ac_space.shape[0]]
        def sample_placeholder(prepend_shape, name=None):
            return tf.placeholder(dtype=sample_dtype, shape=prepend_shape+sample_shape, name=name)

        X = tf.placeholder(tf.float32, ob_shape, name='X') 
        with tf.variable_scope("model", reuse=reuse):

            pi_fc1 = tf.contrib.layers.fully_connected(X, 
                    64, 
                    activation_fn=tf.nn.tanh, 
                    weights_initializer=ortho_init(np.sqrt(2)),
                    scope='pi_fc1')

            pi_fc2 = tf.contrib.layers.fully_connected(pi_fc1, 
                    64, 
                    activation_fn=tf.nn.tanh, 
                    weights_initializer=ortho_init(np.sqrt(2)),
                    scope='pi_fc2')

            pi = tf.contrib.layers.fully_connected(pi_fc2, 
                    actdim, 
                    activation_fn=None, 
                    weights_initializer=ortho_init(0.01),
                    scope='pi')
            
            vf_fc1 = tf.contrib.layers.fully_connected(X, 
                    64, 
                    activation_fn=tf.nn.tanh, 
                    weights_initializer=ortho_init(np.sqrt(2)),
                    scope='vf_fc1')

            vf_fc2 = tf.contrib.layers.fully_connected(vf_fc1, 
                    64, 
                    activation_fn=tf.nn.tanh, 
                    weights_initializer=ortho_init(np.sqrt(2)),
                    scope='vf_fc2')

            vf = tf.contrib.layers.fully_connected(vf_fc2, 
                    1, 
                    activation_fn=None, 
                    weights_initializer=ortho_init(1.0),
                    scope='vf')[:,0]

            logstd = tf.get_variable(name="logstd", shape=[1, actdim],
                initializer=tf.zeros_initializer())

        pd_params = tf.concat([pi, pi * 0.0 + logstd], axis=1)
        self.pd = DiagGaussianPd(pd_params)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)

        def step(ob):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
            return a, v, neglogp

        def value(ob):
            return sess.run(vf, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        self.sample_placeholder = sample_placeholder
