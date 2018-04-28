
import argparse
import os
import gym
from gym.wrappers.monitor import Monitor
import tensorflow as tf
from rlexperiments.ppo import ppo
from rlexperiments.ppo.policies import MlpPolicy
from rlexperiments.vec_env.dummy_vec_env import DummyVecEnv
from rlexperiments.vec_env.vec_normalize import VecNormalize
from rlexperiments.vec_env.episode_monitor import EpisodeMonitor
from rlexperiments.common.tf_util import create_session
from rlexperiments.common.util import ensure_dir


def train(env_id, num_timesteps, seed, cuda_visible_devices, gpu_memory_fraction, output_dir, record_video, record_video_freq = 100):
    if env_id.startswith('Roboschool'):
        import roboschool
    def make_env():
        env = gym.make(env_id)
        if record_video:
            video_path = os.path.join(output_dir, 'video')
            ensure_dir(video_path)
            env = Monitor(env, video_path, video_callable=lambda episode_id: episode_id % record_video_freq == 0, force=True)
        return env    


    env = DummyVecEnv([make_env])
    episode_monitor = EpisodeMonitor(env)
    vec_normalize = VecNormalize(episode_monitor)
    env = vec_normalize

    policy = MlpPolicy

    sess = create_session(cuda_visible_devices, gpu_memory_fraction)

    ppo.learn(policy=policy, env=env, sess=sess, nsteps=2048, nminibatches=32,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=10,
        ent_coef=0.0,
        lr=3e-4,
        cliprange=0.2,
        total_timesteps=num_timesteps,
        vec_normalize=vec_normalize,
        output_dir=output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', default='RoboschoolHopper-v1')
    parser.add_argument('--num-timesteps', help='number of timesteps', type=int, default=int(50e6))
    parser.add_argument('--cuda-visible-devices', help='comma separated list of GPU IDs used by CUDA', default='0')
    parser.add_argument('--gpu-memory-fraction', help='fraction of GPU memory used', type=float, default=0.5)
    parser.add_argument('--output-dir', help='base directory to store models and summaries', default='/tmp/train/ppo')
    parser.add_argument('--video', help='enable video recording', action='store_true')


    args = parser.parse_args()
    train(args.env, num_timesteps=args.num_timesteps, seed=0, cuda_visible_devices=args.cuda_visible_devices, 
        gpu_memory_fraction=args.gpu_memory_fraction, output_dir=args.output_dir, record_video=args.video)


if __name__ == '__main__':
    main()