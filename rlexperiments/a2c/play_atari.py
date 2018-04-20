import time
import os
import argparse
from random import randint
import numpy as np
import tensorflow as tf
import gym  
from rlexperiments.a2c.policy import Policy
from rlexperiments.common.atari_utils import make_atari, wrap_deepmind
from rlexperiments.common.atari_utils import valid_atari_actions

def update_obs(obs, next_obs):
    nc = 1
    obs = np.roll(obs, shift=-nc, axis=3)
    obs[:, :, :, -nc:] = next_obs
    return obs

def run(env_id, model_path):

    env = make_atari(env_id)
    env = wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False)

    num_env = 1
    valid_actions = valid_atari_actions(env, env_id)
    num_actions = len(valid_actions)

    ob_space = env.observation_space
    
    obs = np.zeros((num_env, 84, 84, 4), dtype=np.uint8)
    next_obs = env.reset()    
    obs = update_obs(obs, next_obs)

    steps = 0
    total_reward = 0

    with tf.Session() as sess:

        print('Loading Model %s' % model_path)
        policy = Policy(sess, num_actions, num_env, num_steps=1)


        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

        while True:
            env.render()
            actions, values = policy.step(obs)


            value = values[0]
            steps += 1
            next_obs, rewards, dones, info = env.step(valid_actions[actions[0]])
            total_reward += rewards
            print('%d: reward=%f value=%f' % (steps, total_reward, value))
            obs = update_obs(obs, next_obs)

            if dones:
                print('DONE')
                steps = 0
                total_reward = 0
                next_obs = env.reset()
                obs = np.zeros((num_env, 84, 84, 4), dtype=np.uint8)
                obs = update_obs(obs, next_obs)
                time.sleep(2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', default='PongNoFrameskip-v4')
    parser.add_argument('--model-path', help='model path')
    args = parser.parse_args()
    run(args.env, args.model_path)

if __name__ == "__main__":
    main()
