import time
import re
import os
import argparse
from random import randint
import numpy as np
import gym
import tensorflow as tf
from rlexperiments.ppo.policies import MlpPolicy
from rlexperiments.vec_env.dummy_vec_env import DummyVecEnv
from rlexperiments.vec_env.vec_normalize import VecNormalize
from rlexperiments.vec_env.episode_monitor import EpisodeMonitor

import pyglet, pyglet.window as pw, pyglet.window.key as pwk
from pyglet import gl

class PygletInteractiveWindow(pw.Window):
    def __init__(self, env):
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

    def on_key_press(self, key, modifiers):
        self.keys[key] = +1
        if key==pwk.ESCAPE: self.still_open = False

    def on_key_release(self, key, modifiers):
        self.keys[key] = 0

    def each_frame(self):
        self.theta += 0.05 * (self.keys.get(pwk.LEFT, 0) - self.keys.get(pwk.RIGHT, 0))

def last_vec_norm_file(model_path):
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


def run(env_id, model_path):

    if env_id.startswith('Roboschool'):
        import roboschool

    real_env = gym.make(env_id)

    def make_env():
        return real_env
    env = DummyVecEnv([make_env])
    episode_monitor = EpisodeMonitor(env)
    vec_normalize = VecNormalize(episode_monitor)
    env = vec_normalize

    ob_space = env.observation_space
    ac_space = env.action_space

    control_me = PygletInteractiveWindow(real_env.unwrapped)

    obs = env.reset()    

    steps = 0
    total_reward = 0

    with tf.Session() as sess:

        policy = MlpPolicy(sess, ob_space, ac_space, nbatch=1, nsteps=1)

        print('Loading Model %s' % model_path)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

        vec_norm_state = last_vec_norm_file(model_path)
        print('Loading VecNormalize state %s' % vec_norm_state)
        vec_normalize.restore(vec_norm_state)

        while True:

            #print(env.unwrapped.spec.id)
            #print(real_env) # TimeLimit
            #print(real_env.unwrapped) # RoboschoolHopper
            print(real_env.unwrapped.body_xyz)
    



            actions, values, _ = policy.step(obs)


            img = real_env.render("rgb_array")

            control_me.imshow(img)
            control_me.each_frame()
            if control_me.still_open==False: 
                print('it ends here!!!')
                break


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
