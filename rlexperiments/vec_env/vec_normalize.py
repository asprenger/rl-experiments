from rlexperiments.vec_env import VecEnvWrapper
from rlexperiments.common.running_mean_std import RunningMeanStd
import numpy as np
import pickle

class VecNormalize(VecEnvWrapper):

    """
    Vectorized environment base class
    """
    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step_wait(self):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)
        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        return self._obfilt(obs)

    def snapshot(self, file_path):
        state = {
            'ob_rms': self.ob_rms,
            'ret_rms': self.ret_rms,
            'clipob': self.clipob,
            'cliprew': self.cliprew,
            'ret': self.ret,
            'gamma': self.gamma,
            'epsilon': self.epsilon
        }
        with open(file_path, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    def restore(self, file_path):
        with open(file_path, 'rb') as f:
            state = pickle.load(f)
            print(state)
            self.ob_rms = state['ob_rms']
            self.ret_rms = state['ret_rms']
            self.clipob = state['clipob']
            self.cliprew = state['cliprew']
            self.ret = state['ret']
            self.gamma = state['gamma']
            self.epsilon = state['epsilon']
