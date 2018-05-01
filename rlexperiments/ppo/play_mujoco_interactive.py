import time
import os
import argparse
from random import randint
import numpy as np
import gym
import tensorflow as tf
from rlexperiments.ppo.policies import MlpPolicy
from rlexperiments.ppo.utils import last_vec_norm_path
from rlexperiments.vec_env.dummy_vec_env import DummyVecEnv
from rlexperiments.vec_env.vec_normalize import VecNormalize
from rlexperiments.vec_env.episode_monitor import EpisodeMonitor


def run(env_id, model_path):

    if env_id.startswith('Roboschool'):
        import roboschool

    gym_env = gym.make(env_id)
    def make_env():
        return gym_env
    dummy_vec_env = DummyVecEnv([make_env])
    vec_normalize = VecNormalize(dummy_vec_env)
    vec_env = vec_normalize
    ob_space = vec_env.observation_space
    ac_space = vec_env.action_space

    obs = vec_env.reset()    

    ep = 1
    steps = 0
    total_reward = 0

    with tf.Session() as sess:

        policy = MlpPolicy(sess, ob_space, ac_space, nbatch=1, nsteps=1)

        print('Loading Model %s' % model_path)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

        vec_norm_state = last_vec_norm_path(model_path)
        print('Loading VecNormalize state %s' % vec_norm_state)
        vec_normalize.restore(vec_norm_state)

        while True:
            gym_env.render()
            actions, values, _ = policy.step(obs)

            value = values[0]
            steps += 1

            obs, rewards, dones, info = vec_env.step(actions)
            total_reward += rewards
            print('%d: reward=%f value=%f total_reward=%f' % (steps, rewards[0], value, total_reward))
            
            if dones[0]:
                print('Episode %d finished' % ep)
                ep += 1
                steps = 0
                total_reward = 0
                obs = vec_env.reset()
                time.sleep(2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', default='Hopper-v2')
    parser.add_argument('--model-path', help='model path')
    args = parser.parse_args()
    run(args.env, args.model_path)

if __name__ == "__main__":
    main()
