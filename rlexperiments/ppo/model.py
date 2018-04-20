import numpy as np
import tensorflow as tf


class Model(object):
    def __init__(self, policy, sess, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm):

        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, reuse=True)

        # model inputs
        ACTIONS = train_model.sample_placeholder([None])           # selected actions
        ADVANTAGES = tf.placeholder(tf.float32, [None])            # advantages
        REWARDS = tf.placeholder(tf.float32, [None])               # rewards
        OLD_NEG_LOGP_ACTION = tf.placeholder(tf.float32, [None])   # old negative log action probability
        OLD_VALUE_PRED = tf.placeholder(tf.float32, [None])        # old value prediction
        LR = tf.placeholder(tf.float32, [])                        # learn rate
        CLIP_RANGE = tf.placeholder(tf.float32, [])                # clipping parameter (epsilon)

        # policy gradient loss (clipped surrogate objective)
        neglogpac = train_model.pd.neglogp(ACTIONS)
        ratio = tf.exp(OLD_NEG_LOGP_ACTION - neglogpac) # equivalent to: ratio = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)
        pg_losses = ADVANTAGES * ratio
        pg_losses2 = ADVANTAGES * tf.clip_by_value(ratio, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE)
        pg_loss = tf.reduce_mean(-tf.minimum(pg_losses, pg_losses2))

        # Note about 'pg_loss': the PPO paper proposes the following expression:
        #  argmax_theta min(r_t * A_t, clip() * A_t)
        #
        # Because TF optimizers only minimize functions the expression must be negated.
        #  argmin_theta -min(r_t * A_t, clip() * A_t)

        # value loss
        vpred = train_model.vf
        vpredclipped = OLD_VALUE_PRED + tf.clip_by_value(train_model.vf - OLD_VALUE_PRED, - CLIP_RANGE, CLIP_RANGE)
        vf_losses1 = tf.square(vpred - REWARDS)
        vf_losses2 = tf.square(vpredclipped - REWARDS)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # Adding the entropy of the policy pi to the loss function improves exploration 
        # by discouraging premature convergence to suboptimal deterministic policies.
        # see Async Methods for Deep RL paper (https://arxiv.org/abs/1602.01783)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # final loss
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        # for debugging
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLD_NEG_LOGP_ACTION)) 
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIP_RANGE))) # fraction of timesteps that have been clipped

        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)

        def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8) # TODO extract
            feed_dict = {
                train_model.X: obs, 
                ACTIONS: actions, 
                ADVANTAGES: advs, 
                REWARDS: returns, 
                LR: lr,
                CLIP_RANGE: cliprange, 
                OLD_NEG_LOGP_ACTION: neglogpacs, 
                OLD_VALUE_PRED: values
            }
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                feed_dict
            )[:-1]

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value

        tf.global_variables_initializer().run(session=sess)
