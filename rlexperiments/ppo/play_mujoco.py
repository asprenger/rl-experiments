import time
import os
import argparse
from random import randint
import numpy as np
import tensorflow as tf
import gym
from rlexperiments.ppo.policies import MlpPolicy
from rlexperiments.vec_env.dummy_vec_env import DummyVecEnv
from rlexperiments.vec_env.vec_normalize import VecNormalize
from rlexperiments.vec_env.episode_monitor import EpisodeMonitor

def run(env_id, model_path):

    real_env = gym.make(env_id)

    def make_env():
        return real_env
    env = DummyVecEnv([make_env])
    episode_monitor = EpisodeMonitor(env)
    vec_normalize = VecNormalize(episode_monitor)
    env = vec_normalize

    ob_space = env.observation_space
    ac_space = env.action_space

    obs = env.reset()    

    steps = 0
    total_reward = 0

    with tf.Session() as sess:

        print('Loading Model %s' % model_path)

        policy = MlpPolicy(sess, ob_space, ac_space, nbatch=1, nsteps=1)

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

        vec_normalize.restore('/tmp/train/ppo/model/vec_normalize-2850.pickle')

        while True:
            real_env.render()
            actions, values, _ = policy.step(obs)

            value = values[0]
            steps += 1

            obs, rewards, dones, info = env.step(actions)
            total_reward += rewards
            print('%d: reward=%f value=%f' % (steps, total_reward, value))
            
            if dones:
                print('ep_reward=%f ep_length=%f' % (episode_monitor.mean_episode_reward(), episode_monitor.mean_episode_length()))
                print('DONE')
                steps = 0
                total_reward = 0
                obs = env.reset()
                time.sleep(2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', default='Hopper-v2')
    parser.add_argument('--model-path', help='model path')
    args = parser.parse_args()
    run(args.env, args.model_path)

if __name__ == "__main__":
    main()
