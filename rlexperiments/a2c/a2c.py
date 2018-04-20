'''
Advantage Actor Critic (A2C)
'''

import os
import tensorflow as tf
from rlexperiments.a2c.model import Model
from rlexperiments.a2c.runner import Runner
from rlexperiments.common.tf_util import create_session
from rlexperiments.common.util import ensure_dir
from rlexperiments.common.atari_utils import valid_atari_actions


def learn(env, env_id, num_env, total_timesteps, output_dir, cuda_visible_devices, gpu_memory_fraction, load_model):

    valid_actions = valid_atari_actions(env, env_id)
    num_actions = len(valid_actions)

    report_summary_freq = 100
    save_model_freq = 2000
    
    num_steps = 5
    batch_size = num_env * num_steps

    model_path = os.path.join(output_dir, 'model')
    summary_path = os.path.join(output_dir, 'summary')

    ensure_dir(summary_path)
    ensure_dir(model_path)

    sess = create_session(cuda_visible_devices, gpu_memory_fraction)

    model = Model(sess, num_env, num_steps, num_actions, total_timesteps=total_timesteps)
    runner = Runner(env, num_env, model, valid_actions, num_steps=num_steps)

    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(summary_path)

    if load_model:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)


    timesteps = 0
    max_updates = total_timesteps // batch_size
    print("Number of updates: %d" % max_updates)

    for update in range(1, max_updates+1):

        obs, rewards, actions, values = runner.run()

        timesteps = update * batch_size

        policy_loss, value_loss, policy_entropy, cur_lr = model.train(obs, rewards, actions, values)

        if update % report_summary_freq == 0 and update != 0:

            mean_reward = safe_mean(runner.running_rewards.copy())
            mean_steps = safe_mean(runner.running_steps.copy())
            mean_value = safe_mean(runner.running_values.copy())

            print("Updates: %d" % update)
            print("Timesteps: %d" % timesteps)
            print("Learn rate: %f" % cur_lr)
            print("Policy loss: %f" % float(policy_loss))
            print("Value loss: %f" % float(value_loss))
            print("Running rewards: %s" % runner.running_rewards)
            print("Mean reward: %s" % mean_reward)
            print("Mean steps: %s" % mean_steps)
            print("Mean values: %s" % mean_value)

            train_summary = tf.Summary()
            train_summary.value.add(tag='Train/Timesteps', simple_value=timesteps)
            train_summary.value.add(tag='Train/Policy loss', simple_value=policy_loss)
            train_summary.value.add(tag='Train/Policy entropy', simple_value=policy_entropy)
            train_summary.value.add(tag='Train/Value loss', simple_value=value_loss)
            train_summary.value.add(tag='Train/Mean steps', simple_value=mean_steps)
            train_summary.value.add(tag='Train/Mean reward', simple_value=mean_reward)
            train_summary.value.add(tag='Train/Mean values', simple_value=mean_value)
            summary_writer.add_summary(train_summary, update)
            summary_writer.flush()    

        if update % save_model_freq == 0 and update != 0:
            print('Save model')
            saver.save(sess, model_path + '/model-' + str(update) + '.ckpt')

def safe_mean(values):
    values[values == None] = 0.0
    return values.mean()

