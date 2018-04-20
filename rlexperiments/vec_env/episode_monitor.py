import numpy as np
from rlexperiments.vec_env import VecEnvWrapper


class EpisodeMonitor(VecEnvWrapper):

    def __init__(self, venv):
        VecEnvWrapper.__init__(self, venv)
        self.nenv = venv.num_envs
        self.episode_rewards = np.zeros((self.nenv,))
        self.episode_lengths = np.zeros((self.nenv,))
        self.mean_episode_rewards = np.full((self.nenv,), None)
        self.mean_episode_lengths = np.full((self.nenv,), None)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.episode_rewards += rewards
        self.episode_lengths += np.ones(self.nenv)
        for i, done in enumerate(dones):
            if done:
                self.mean_episode_rewards[i] = self.episode_rewards[i] if self.mean_episode_rewards[i] is None else self.mean_episode_rewards[i] * 0.99 + self.episode_rewards[i] * 0.01
                self.episode_rewards[i] = 0
                self.mean_episode_lengths[i] = self.episode_lengths[i] if self.mean_episode_lengths[i] is None else self.mean_episode_lengths[i] * 0.99 + self.episode_lengths[i] * 0.01
                self.episode_lengths[i] = 0
        for i, info in enumerate(infos):
            info['episode.mean_reward'] = self.mean_episode_rewards[i]
            info['episode.mean_length'] = self.mean_episode_lengths[i]
        return obs, rewards, dones, infos

    def reset(self):
        # TODO fix reset logic
        obs = self.venv.reset()
        self.episode_rewards = np.zeros((self.nenv,))
        self.episode_lengths = np.zeros((self.nenv,))
        return obs

    def mean_episode_reward(self):
        return self._safe_mean(self.mean_episode_rewards.copy())
        
    def mean_episode_length(self):
        return self._safe_mean(self.mean_episode_lengths.copy())

    def _safe_mean(self, values):
        values[values == None] = 0.0
        return values.mean()
    