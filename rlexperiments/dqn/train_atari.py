'''
Pong Deep-Q Learning (DQN)
'''

import numpy as np
import argparse
import os
import random
import gym
from gym.wrappers.monitor import Monitor
import tensorflow as tf
from rlexperiments.common.tf_util import create_session
from rlexperiments.common.atari_utils import make_atari, wrap_deepmind
from rlexperiments.dqn.dqn import learn

def make_atari_env(env_id, seed, output_dir, record_video, record_video_freq = 100):
    env = make_atari(env_id)
    env.seed(seed)
    if record_video:
        video_path = os.path.join(output_dir, 'video/env-%d' % rank)
        ensure_dir(video_path)
        env = Monitor(env, video_path, video_callable=lambda episode_id: episode_id % record_video_freq == 0, force=True)
    return wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=True)

def train(env_id, cuda_visible_devices, gpu_memory_fraction, output_dir):
    seed = random.randint(0, 1e6)
    env = make_atari_env(env_id, seed, output_dir, record_video=False)
    sess = create_session(cuda_visible_devices, gpu_memory_fraction)
    learn(env, sess, cuda_visible_devices, gpu_memory_fraction, output_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', default='PongNoFrameskip-v4')
    parser.add_argument('--cuda-visible-devices', help='comma separated list of GPU IDs used by CUDA', default='0')
    parser.add_argument('--gpu-memory-fraction', help='fraction of GPU memory used', type=float, default=0.5)
    parser.add_argument('--output-dir', help='base directory to store models and summaries', default='/tmp/train/ppo')
    args = parser.parse_args()
    train(args.env, cuda_visible_devices=args.cuda_visible_devices, 
        gpu_memory_fraction=args.gpu_memory_fraction, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
