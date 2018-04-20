
import os
import numpy as np
import tensorflow as tf
from rlexperiments.common.util import ensure_dir
from rlexperiments.common.replay_buffer import ReplayBuffer
from rlexperiments.common.schedules import LinearSchedule
from rlexperiments.dqn.policy import Qnetwork
from rlexperiments.dqn.runner import Runner

# TODO:
# replace MSE loss function, see:

#   https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/ 
#   https://www.reddit.com/r/MachineLearning/comments/6dmzdy/d_on_using_huber_loss_in_deep_qlearning/
#   https://jaromiru.com/2016/10/21/lets-make-a-dqn-full-dqn/
#   https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b

# use gradient clipping
# use learn schedule


def update_target(from_scope, to_scope):
    '''Returns an operation that copies the trainable variables from the 
       'from_scope' to the trainable variables in the 'to_scope'. It is
       assumed that the two scopes have the exact same trainable variables.
    '''
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    assert len(from_vars) == len(to_vars)
    update_target_ops = []
    for from_var, to_var in zip(sorted(from_vars, key=lambda v: v.name),
                                sorted(to_vars, key=lambda v: v.name)):
        update_target_ops.append(to_var.assign(from_var))
    return tf.group(*update_target_ops)


def learn(env, sess, cuda_visible_devices, gpu_memory_fraction, output_dir):


    log_freq = 100
    save_freq = 500
    update_target_freq = 250

    total_timesteps = int(50e6)

    batch_size = 32 # How many experiences to use for each training step.

    exploration_start_eps = 1.0    # Starting probability of random action
    exploration_final_eps = 0.02   # End probability of random action
    exploration_max_steps = 200000 # How many steps of training to reduce epsilon

    hidden_size = 256
    pre_train_steps = 5000 
    
    nsteps = 4        # How often to perform a training step.
    gamma = 0.99           # Discount factor on the target Q-values

    render = False
    load_model = False

    model_path = os.path.join(output_dir, 'model')
    summary_path = os.path.join(output_dir, 'summary')
    video_path = os.path.join(output_dir, 'video')
    ensure_dir(summary_path)
    ensure_dir(model_path)
    ensure_dir(video_path)

    valid_actions = [2, 3] # TODO replace
    num_actions = len(valid_actions)

    

    mainQN = Qnetwork(hidden_size, num_actions, "main", add_summaries=True)
    targetQN = Qnetwork(hidden_size, num_actions, "target")

    saver = tf.train.Saver()

    # operation that copies a snapshot of the main network to the target network
    update_target_op = update_target("main", "target")

    replay_buffer = ReplayBuffer(50000)

    # Create an exploration schedule 
    exploration = LinearSchedule(schedule_timesteps=exploration_max_steps,
                                 initial_p=exploration_start_eps,
                                 final_p=exploration_final_eps)

    with sess:
        
        sess.run(tf.global_variables_initializer())
        
        summary_writer = tf.summary.FileWriter(summary_path)
        
        if load_model:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

        print("Populating replay buffer")
        state = env.reset()
        i = 0
        while(True):
            action = np.random.randint(0, num_actions)
            next_state, reward, done, _ = env.step(valid_actions[action])
            replay_buffer.add(state, action, reward, next_state, float(done))
            if done:
                state = env.reset()
            else:
                state = next_state
            i += 1
            if i > pre_train_steps and done:
                break    
                
        runner = Runner(sess, env, replay_buffer, num_actions, nsteps, exploration, mainQN, valid_actions)

        nupdates = total_timesteps // nsteps

        for update in range(1, nupdates + 1):

            runner.run()
            
            if update % update_target_freq == 0:
                sess.run(update_target_op)

            # Sample a batch of transitions from the replay buffer
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

            # Calculate the maximizing action q-value for s_tp1 using 'Double Q-learning'

            # 1. Predict the action that maximizes the q-value for s_tp1 using the mainQN
            feed_dict = {
                mainQN.state_input: next_states # shape: (batch_size, 84, 84, 4)
            }
            Q1 = sess.run(mainQN.max_q_action, feed_dict=feed_dict) # shape: (batch_size,)

            # 2. Predict the q-values for s_tp1 using the targetQN
            feed_dict = {
                targetQN.state_input: next_states
            }
            Q2 = sess.run(targetQN.q_out, feed_dict=feed_dict) # (batch_size, 2)

            # 3. Get the maxiziming action q-value for s_tp1 by selecting the Q1 index from the Q2 array
            max_action_q = Q2[range(batch_size), Q1] # (batch_size,)

            # inverte the 'done' fields in the train_batch, e.g. 000010000 -> 111101111
            # in 'done' transitions there are no future rewards and the update rule reduces to: target_q = reward
            inverted_done_indicator = -(dones - 1)

            # Calculate the target-q value, that is what we think is the correct q-value for s_t and the 
            # selected action and is used to calculate the td-error.
            target_q = rewards + (gamma * max_action_q * inverted_done_indicator)

            # Update the mainQN
            feed_dict = {
                mainQN.state_input: states,
                mainQN.actions: actions,
                mainQN.target_q: target_q
            }
            _, summaries = sess.run([mainQN.train_op, mainQN.summaries], feed_dict)


            if update % log_freq == 0:
                print('update %d: mean_ep_reward=%f, mean_ep_length=%d, total_steps=%d' % (update, runner.mean_ep_reward or 0.0, runner.mean_ep_length or 0.0, runner.total_steps))
                print('Report summaries')
                train_summary = tf.Summary()
                train_summary.value.add(tag='Train/Episode Reward', simple_value=runner.mean_ep_reward)
                train_summary.value.add(tag='Train/Episode Length', simple_value=runner.mean_ep_length)
                summary_writer.add_summary(train_summary, update)
                summary_writer.add_summary(summaries, update)
                summary_writer.flush()

            if update % save_freq == 0:
                print('Save model')
                saver.save(sess, model_path + '/model-' + str(update) + '.ckpt')
