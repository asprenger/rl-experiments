import tensorflow as tf
from rlexperiments.common.scheduler import Scheduler
from rlexperiments.common.tf_util import create_session
from rlexperiments.a2c.policy import Policy


class Model(object):

    def __init__(self, sess, num_env, num_steps, num_actions, ent_coef=0.01, vf_coef=0.5, 
            max_grad_norm=0.5, alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), 
            lrschedule='linear', lr=7e-4):

        num_batch = num_env * num_steps

        A = tf.placeholder(tf.int32, [num_batch])     # selected actions
        ADV = tf.placeholder(tf.float32, [num_batch]) # adventage of taking the action
        R = tf.placeholder(tf.float32, [num_batch])   # discounted rewards of taking the action
        LR = tf.placeholder(tf.float32, [])           # learning rate

        # step model takes num_env states and calculates actions and values
        step_model = Policy(sess, num_actions, num_env, 1, reuse=False)

        # train model takes num_env*num_steps observations and updates model parameters
        train_model = Policy(sess, num_actions, num_env, num_steps, reuse=True)

        # Note: step_model and train_model share the same network instance, the difference 
        # is just the input shape because they are used with different batch sizes

        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)
        pg_loss = tf.reduce_mean(ADV * neglogpac)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        entropy = tf.reduce_mean(train_model.pd.entropy())

        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        params = find_trainable_variables('model')
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)    

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, rewards, actions, values):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()
            td_map = {
                train_model.X: obs, 
                A: actions, 
                ADV: advs, 
                R: rewards, 
                LR: cur_lr
            }

            policy_loss, value_loss, policy_entropy, _ = sess.run([pg_loss, vf_loss, entropy, _train], td_map)

            return policy_loss, value_loss, policy_entropy, cur_lr

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value

        sess.run(tf.global_variables_initializer())

def mse(pred, target):
    return tf.square(pred - target) / 2.0

def find_trainable_variables(key):
    with tf.variable_scope(key):
        return tf.trainable_variables()
