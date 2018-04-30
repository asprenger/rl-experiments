import time
import re
import os
import argparse
from random import randint
import numpy as np
import pyglet, pyglet.window as pw, pyglet.window.key as pwk
from pyglet import gl
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import tensorflow as tf
from rlexperiments.common.util import ts_rand
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


def last_vec_norm_path(model_path):
    if not os.path.exists(model_path):
        return None
    files = os.listdir(model_path)
    files = [file for file in files if file.startswith('vec_normalize-')]
    if len(files) == 0:
        return None
    epochs = [[i, int(re.search('vec_normalize-(.*).pickle', f).group(1))] for i,f in enumerate(files)]
    epochs.sort(key=lambda x: -x[1])
    last_epoch_idx = epochs[0][0]
    return os.path.join(model_path, files[last_epoch_idx])


def ep_video_path(video_path, ts, env_id, epoch):
    return os.path.join(video_path, '%s-%s-%d.mp4' % (ts, env_id, epoch))

def run(env_id, model_path, video_path=None):

    if env_id.startswith('Roboschool'):
        import roboschool

    record_video = video_path != None

    gym_env = gym.make(env_id)

    def make_env():
        return gym_env
    dummy_venv = DummyVecEnv([make_env])
    ep_monitor = EpisodeMonitor(dummy_venv)
    vec_normalize = VecNormalize(ep_monitor)
    venv = vec_normalize

    ob_space = venv.observation_space
    ac_space = venv.action_space

    window = PygletWindow()

    obs = venv.reset()    

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
        video_recorder = VideoRecorder(gym_env, path=ep_video_path(video_path, ts, env_id, ep), enabled=record_video)

        while True:    
            actions, values, _ = policy.step(obs)
            img = gym_env.render("rgb_array")
            window.imshow(img)
            video_recorder.capture_frame()

            if window.still_open==False:
                video_records.close()
                break

            value = values[0]
            steps += 1
            obs, rewards, dones, info = venv.step(actions)
            total_reward += rewards
            print('%d: reward=%f value=%f' % (steps, total_reward, value))
            if dones:
                print('ep_reward=%f ep_length=%f\nDONE' % (ep_monitor.mean_episode_reward(), ep_monitor.mean_episode_length()))
                ep += 1
                steps = 0
                total_reward = 0
        
                #window.close()
                #window = PygletWindow()

                video_recorder.close()
                video_recorder = VideoRecorder(gym_env, path=ep_video_path(video_path, ts, env_id, ep), enabled=record_video)
                obs = venv.reset()
                time.sleep(2)
                

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='Gym environment ID', default='Hopper-v2')
    parser.add_argument('--model-path', help='TensorFlow model path')
    parser.add_argument('--video-path', help='video path')
    args = parser.parse_args()
    run(args.env, args.model_path, args.video_path)

if __name__ == "__main__":
    main()
