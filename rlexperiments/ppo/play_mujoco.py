import time
import os
import argparse
from random import randint
import numpy as np
import pyglet, pyglet.window as pw, pyglet.window.key as pwk
from pyglet import gl
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import tensorflow as tf
from rlexperiments.common.util import ts_rand, ensure_dir
from rlexperiments.ppo.utils import last_vec_norm_path
from rlexperiments.ppo.policies import MlpPolicy
from rlexperiments.vec_env.dummy_vec_env import DummyVecEnv
from rlexperiments.vec_env.vec_normalize import VecNormalize
from rlexperiments.vec_env.episode_monitor import EpisodeMonitor

class PygletWindow(pw.Window):
    def __init__(self):
        pw.Window.__init__(self, width=600, height=400, vsync=False, resizable=True)
        self.theta = 0
        self.still_open = True

        @self.event
        def on_close():
            self.still_open = False

        @self.event
        def on_resize(width, height):
            self.win_w = width
            self.win_h = height

        self.keys = {}
        self.human_pause = False
        self.human_done = False


    def imshow(self, arr):
        H, W, C = arr.shape
        assert C==3
        image = pyglet.image.ImageData(W, H, 'RGB', arr.tobytes(), pitch=W*-3)
        self.clear()
        self.switch_to()    
        self.dispatch_events()
        texture = image.get_texture()
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        texture.width  = W
        texture.height = H
        texture.blit(0, 0, width=self.win_w, height=self.win_h)
        self.flip()


def ep_video_path(video_path, ts, env_id, epoch):
    return os.path.join(video_path, '%s-%s-%d.mp4' % (ts, env_id, epoch))


def run(env_id, model_path, record_video, video_path=None):

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

    window = PygletWindow()

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

        vec_norm_path = last_vec_norm_path(model_path)
        print('Loading VecNormalize state %s' % vec_norm_path)
        vec_normalize.restore(vec_norm_path)

        ts = ts_rand()
        if record_video:
            ensure_dir(video_path)
            video_recorder = VideoRecorder(gym_env, path=ep_video_path(video_path, ts, env_id, ep))

        while True:    
            actions, values, _ = policy.step(obs)

            img = gym_env.render("rgb_array")
            window.imshow(img)
            
            if record_video:
                video_recorder.capture_frame()

            if window.still_open==False:
                video_records.close()
                break

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

                window.close()
                window = PygletWindow()

                if record_video:
                    video_recorder.close()
                    video_recorder = VideoRecorder(gym_env, path=ep_video_path(video_path, ts, env_id, ep), enabled=record_video)
                obs = vec_env.reset()
                time.sleep(2)
                

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='Gym environment ID', default='Hopper-v2')
    parser.add_argument('--model-path', help='TensorFlow model path')
    parser.add_argument('--video', help='enable video recording', action='store_true')
    parser.add_argument('--video-path', help='video path', default='/tmp/video')
    args = parser.parse_args()
    run(args.env, args.model_path, args.video, args.video_path)

if __name__ == "__main__":
    main()
