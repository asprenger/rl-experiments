import os
import argparse
from random import randint
from rlexperiments.common.util import ts_rand
from rlexperiments.common.atari_utils import make_atari, wrap_deepmind
from rlexperiments.vec_env.subproc_vec_env import SubprocVecEnv
from rlexperiments.vec_env.vec_frame_stack import VecFrameStack
from rlexperiments.a2c.a2c import learn

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
    return SubprocVecEnv([make_env(i) for i in range(num_env)])


def run(env_id, num_env=8, total_timesteps=int(50e6), base_dir='/tmp/train/a2c', record_video=False, 
        cuda_visible_devices='0', gpu_memory_fraction=0.5, load_model=False):
    output_dir = os.path.join(base_dir, ts_rand())
    print('Output dir: %s' % output_dir)
    env = make_atari_env(env_id, num_env, seed = randint(0, 1000000), output_dir=output_dir, record_video=record_video)
    # Do frame-stacking here instead of the FrameStack wrapper to reduce IPC overhead
    env = VecFrameStack(env, 4)
    learn(env, env_id, num_env=num_env, total_timesteps=total_timesteps, output_dir=output_dir, cuda_visible_devices=cuda_visible_devices, 
          gpu_memory_fraction=gpu_memory_fraction, load_model=load_model)
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', default='PongNoFrameskip-v4')
    parser.add_argument('--num-env', help='number of environments', type=int, default=8)
    parser.add_argument('--num-timesteps', help='number of timesteps', type=int, default=int(50e6))
    parser.add_argument('--output-dir', help='base directory to store models and summaries', default='/tmp/train/a2c')
    parser.add_argument('--video', help='Enable video recording', action='store_true')
    parser.add_argument('--cuda-visible-devices', help='comma separated list of GPU IDs used by CUDA', default='0')
    parser.add_argument('--gpu-memory-fraction', help='fraction of GPU memory used', type=float, default=0.5)

    args = parser.parse_args()
    run(env_id=args.env, num_env=args.num_env, total_timesteps=args.num_timesteps, base_dir=args.output_dir, 
        record_video=args.video, cuda_visible_devices=args.cuda_visible_devices, gpu_memory_fraction=args.gpu_memory_fraction)

if __name__ == "__main__":
    main()
