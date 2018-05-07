import os
import argparse
import tensorflow as tf
from gym.wrappers.monitor import Monitor
from rlexperiments.common.atari_utils import make_atari, wrap_deepmind
from rlexperiments.common.tf_util import create_session
from rlexperiments.common.util import ensure_dir
from rlexperiments.vec_env.subproc_vec_env import SubprocVecEnv
from rlexperiments.vec_env.vec_frame_stack import VecFrameStack
from rlexperiments.vec_env.episode_monitor import EpisodeMonitor
from rlexperiments.trpo.policies import CnnPolicy
from rlexperiments.trpo import trpo


def make_atari_env(env_id, num_env, seed, output_dir, record_video, record_video_freq = 100):
    '''
    Create a VecEnv for Atari.
    '''
    def make_env(rank):
        def _thunk():
            env = make_atari(env_id)
            env.seed(seed + rank)
            if record_video:
                video_path = os.path.join(output_dir, 'video/env-%d' % rank)
                ensure_dir(video_path)
                env = Monitor(env, video_path, video_callable=lambda episode_id: episode_id % record_video_freq == 0, force=True)
            return wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False)
        return _thunk
    venv = SubprocVecEnv([make_env(i) for i in range(num_env)])    
    return EpisodeMonitor(venv)


def train(env_id, num_envs, num_timesteps, seed, cuda_visible_devices, gpu_memory_fraction, output_dir, video):
    env = make_atari_env(env_id, num_envs, seed, output_dir=output_dir, record_video=video)
    env = VecFrameStack(env, 4)
    policy = CnnPolicy
    sess = create_session(cuda_visible_devices, gpu_memory_fraction)
    trpo.learn(env, policy, sess, timesteps_per_batch=512, max_kl=0.001, cg_iters=10, cg_damping=1e-3,
        max_timesteps=int(num_timesteps * 1.1), gamma=0.98, lam=1.0, vf_iters=3, vf_stepsize=1e-4, entcoeff=0.00)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', default='PongNoFrameskip-v4')
    parser.add_argument('--num-envs', help='number of environments', type=int, default=8)
    parser.add_argument('--num-timesteps', help='number of timesteps', type=int, default=int(50e6))
    parser.add_argument('--cuda-visible-devices', help='comma separated list of GPU IDs used by CUDA', default='0')
    parser.add_argument('--gpu-memory-fraction', help='fraction of GPU memory used', type=float, default=0.5)
    parser.add_argument('--output-dir', help='base directory to store models and summaries', default='/tmp/train/trpo')
    parser.add_argument('--video', help='enable video recording', action='store_true')

    args = parser.parse_args()
    train(args.env, num_envs=args.num_envs, num_timesteps=args.num_timesteps, seed=0, cuda_visible_devices=args.cuda_visible_devices, 
        gpu_memory_fraction=args.gpu_memory_fraction, output_dir=args.output_dir, video=args.video)

if __name__ == '__main__':
    main()
