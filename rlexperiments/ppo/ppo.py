import os
import time
import numpy as np
import os.path as osp
import tensorflow as tf
from collections import deque
from rlexperiments.common.util import ensure_dir
from rlexperiments.common.tf_util import explained_variance
from rlexperiments.ppo.model import Model
from rlexperiments.ppo.runner import Runner


def learn(policy, env, nsteps, sess, total_timesteps, ent_coef, lr,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=50, cuda_visible_devices='0', gpu_memory_fraction=0.5,
            output_dir=None, vec_normalize=None):

    # TODO DRY
    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    model = Model(policy=policy, sess=sess, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm)

    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    model_path = os.path.join(output_dir, 'model')
    summary_path = os.path.join(output_dir, 'summary')
    ensure_dir(summary_path)
    ensure_dir(model_path)

    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(summary_path)

    tfirststart = time.time()

    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)
        obs, returns, masks, actions, values, neglogpacs, ep_info = runner.run()

        mblossvals = []


        inds = np.arange(nbatch)
        for _ in range(noptepochs):
            np.random.shuffle(inds)
            for start in range(0, nbatch, nbatch_train):
                end = start + nbatch_train
                mbinds = inds[start:end]
                slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                mblossvals.append(model.train(lrnow, cliprangenow, *slices))


        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))

        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            print('')
            print("nupdates", update)
            print("serial_timesteps", update*nsteps)
            print("total_timesteps", update*nbatch)
            print("fps", fps)
            print("explained_variance", float(ev))
            print('mean_episode_reward', ep_info['ep_mean_reward'])
            print('mean_episode_length', ep_info['ep_mean_length'])
            print('time_elapsed', tnow - tfirststart)
            
            policy_loss = lossvals[0]
            value_loss = lossvals[1]
            policy_entropy = lossvals[2]
            approxkl = lossvals[3]
            clipfrac = lossvals[4]

            print("policy_loss", policy_loss)
            print("value_loss", value_loss)

            train_summary = tf.Summary()
            train_summary.value.add(tag='Train/Episode Reward', simple_value=ep_info['ep_mean_reward'])
            train_summary.value.add(tag='Train/Episode Length', simple_value=ep_info['ep_mean_length']) 
            train_summary.value.add(tag='Train/FPS', simple_value=fps)
            train_summary.value.add(tag='Train/Policy Loss', simple_value=policy_loss)
            train_summary.value.add(tag='Train/Value Loss', simple_value=value_loss)
            summary_writer.add_summary(train_summary, update)
            summary_writer.flush()    

        if update % save_interval == 0 and update != 0:
            print('Save model: %s' % model_path)
            saver.save(sess, os.path.join(model_path, 'model-'+str(update)+'.ckpt'))
            if vec_normalize:
                # save observation scaling inside VecNormalize
                vec_normalize.snapshot(os.path.join(model_path, 'vec_normalize-'+str(update)+'.pickle'))

    env.close()

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def constfn(val):
    def f(_):
        return val
    return f
