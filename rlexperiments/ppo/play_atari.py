import time
import os
import argparse
from random import randint
import numpy as np
import tensorflow as tf
import gym  
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from rlexperiments.ppo.policies import CnnPolicy
from rlexperiments.common.atari_utils import make_atari, wrap_deepmind
from rlexperiments.common.util import ts_rand, ensure_dir

# Note: we can not use VecFrameStack because it does not support rendering. 
# Therefore we perform frame-stacking in update_obs().

def update_obs(obs, next_obs):
    nc = 1
    obs = np.roll(obs, shift=-nc, axis=3)
    obs[:, :, :, -nc:] = next_obs
    return obs

def ep_video_path(video_path, ts, env_id, epoch):
    return os.path.join(video_path, '%s-%s-%d.mp4' % (ts, env_id, epoch))

def run(env_id, model_path, record_video, video_path=None):

    env = make_atari(env_id)
    env = wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False)

    num_env = 1
    ob_space = env.observation_space
    ac_space = env.action_space
    
    obs = np.zeros((num_env, 84, 84, 4), dtype=np.uint8)
    next_obs = env.reset()
    obs = update_obs(obs, next_obs)

    ep = 1
    steps = 0
    total_reward = 0

    with tf.Session() as sess:

        print('Loading Model %s' % model_path)
        policy = CnnPolicy(sess, ob_space, ac_space, nbatch=1, nsteps=1)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

        ts = ts_rand()
        if record_video:
            ensure_dir(video_path)
            video_recorder = VideoRecorder(env, path=ep_video_path(video_path, ts, env_id, ep))

        while True:
            env.render()

            if record_video:
                video_recorder.capture_frame()

            actions, values, _ = policy.step(obs)
            value = values[0]
            steps += 1
            next_obs, rewards, dones, info = env.step(actions)
            total_reward += rewards
            print('%d: reward=%f value=%f' % (steps, total_reward, value))
            obs = update_obs(obs, next_obs)

            if dones:
                print('DONE')
                ep += 1
                steps = 0
                total_reward = 0
                next_obs = env.reset()
                obs = np.zeros((num_env, 84, 84, 4), dtype=np.uint8)
                obs = update_obs(obs, next_obs)

                if record_video:
                    video_recorder.close()
                    video_recorder = VideoRecorder(env, path=ep_video_path(video_path, ts, env_id, ep), enabled=record_video)
                '`'
                time.sleep(2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', default='PongNoFrameskip-v4')
    parser.add_argument('--model-path', help='model path')
    parser.add_argument('--video', help='enable video recording', action='store_true')
    parser.add_argument('--video-path', help='video path', default='/tmp/video')
    args = parser.parse_args()
    run(args.env, args.model_path, args.video, args.video_path)

if __name__ == "__main__":
    main()
