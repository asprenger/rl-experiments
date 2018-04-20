import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from rlexperiments.common.util import ortho_init

def layer_activation_summary(name, tensor):
    tf.summary.histogram('Activation/'+name, tensor)
    tf.summary.histogram('Sparsity/'+name, tf.nn.zero_fraction(tensor))


class Qnetwork():

    def __init__(self, hidden_size, num_actions, scope, add_summaries=False):

        with tf.variable_scope(scope):

            self.state_input = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.int8, name="state_input")

            self.conv1 = layers.convolution2d(tf.cast(self.state_input, tf.float32) / 255.0,
                            num_outputs=32,
                            kernel_size=8,
                            stride=4,
                            activation_fn=tf.nn.relu,
                            weights_initializer=ortho_init(np.sqrt(2)),
                            scope='conv1')
        
            self.conv2 = layers.convolution2d(self.conv1,
                            num_outputs=64,
                            kernel_size=4,
                            stride=2,
                            activation_fn=tf.nn.relu,
                            weights_initializer=ortho_init(np.sqrt(2)),
                            scope='conv2')

            self.conv3 = layers.convolution2d(self.conv2,
                            num_outputs=64,
                            kernel_size=3,
                            stride=1,
                            activation_fn=tf.nn.relu,
                            weights_initializer=ortho_init(np.sqrt(2)),
                            scope='conv3')

            self.flattened = layers.flatten(self.conv3)

            self.fc1 = tf.contrib.layers.fully_connected(self.flattened, 
                hidden_size, 
                activation_fn=tf.nn.relu, 
                weights_initializer=ortho_init(np.sqrt(2)),
                scope='fc1')

            # Split the output into separate advantage and value streams.
                        
            # Advantage stream
            self.advantage = tf.contrib.layers.fully_connected(self.fc1, 
                num_actions, 
                activation_fn=None, 
                biases_initializer=None,
                scope='advantage')

            # Value stream
            self.value = tf.contrib.layers.fully_connected(self.fc1, 
                1, 
                activation_fn=None, 
                biases_initializer=None,
                scope='value')

            # Then combine them together to get the predicted q-values for all actions
            self.q_out = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keep_dims=True))

            # Action that maximizes the q-value
            self.max_q_action = tf.argmax(self.q_out, axis=1)
            
            # The selected actions
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32) # shape: (32,)
            self.actions_onehot = tf.one_hot(self.actions, num_actions, dtype=tf.float32)

            # Predicted q-values of the selected actions
            self.selected_actions_q = tf.reduce_sum(tf.multiply(self.q_out, self.actions_onehot), axis=1)

            self.target_q = tf.placeholder(shape=[None], dtype=tf.float32) # shape: (32,)
            self.td_error = self.selected_actions_q - self.target_q

            # Note: in general we have to make sure that during training only the gradients of
            # Q and not targetQ are updated. (See 33:25: http://videolectures.net/deeplearning2017_van_hasselt_deep_reinforcement)
            # In the current implementation this is guaranteed because targetQ is a placeholder
            # and set from the outside. Otherwise we would need to stop the gradient update:
            #
            #   self.td_error = self.selected_actions_q - tf.stop_gradient(self.targetQ)

            self.loss = tf.reduce_mean(tf.square(self.td_error))
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00025, decay=0.99, momentum=0.0, epsilon=1e-6)
            self.train_op = self.optimizer.minimize(self.loss)

        if add_summaries:
            layer_activation_summary('conv1', self.conv1),
            layer_activation_summary('conv2', self.conv2),
            layer_activation_summary('conv3', self.conv3),
            layer_activation_summary('fc1', self.fc1),
            tf.summary.scalar('Train/Loss', self.loss),
            tf.summary.scalar('Train/Action Value', tf.reduce_mean(self.selected_actions_q))
            self.summaries = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))
        else:
            self.summaries = None