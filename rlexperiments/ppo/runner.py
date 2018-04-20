import numpy as np
import tensorflow as tf

class Runner(object):

    def __init__(self, *, env, model, nsteps, gamma, lam):
        self.env = env
        self.model = model
        nenv = env.num_envs
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=model.train_model.X.dtype.name)
        self.obs[:] = env.reset()
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        
    def run(self):

        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        
        for _ in range(self.nsteps):

            actions, values, neglogpacs = self.model.step(self.obs)

            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            
            self.obs[:], rewards, dones, infos = self.env.step(actions)

            mb_rewards.append(rewards)
            mb_dones.append(dones)

        ep_mean_reward = np.array([info['episode.mean_reward'] or 0.0 for info in infos]).mean()
        ep_mean_length = np.array([info['episode.mean_length'] or 0.0 for info in infos]).mean()
        ep_info = {'ep_mean_reward':ep_mean_reward, 'ep_mean_length':ep_mean_length}

        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)

        # estimate state values for the last observations (for GAE)
        last_values = self.model.value(self.obs)

        # estimate advantages using generalized advantage estimation (GAE)
        mb_advs = gae(mb_rewards, mb_values, mb_dones, last_values, self.gamma, self.lam)

        # instead of using the emperical rewards 'mb_rewards' we derive the rewards 
        # from the estimated advantages and values    
        mb_returns = mb_advs + mb_values

        return sf01(mb_obs), sf01(mb_returns), sf01(mb_dones), sf01(mb_actions), sf01(mb_values), sf01(mb_neglogpacs), ep_info


def gae(rewards, values, dones, last_values, gamma, lam):
    '''
    Calculate generalized advantage estimation for a rollout

    rewards : array (num_steps, nenv)
    values : array (num_steps, nenv)
    dones : array (num_steps, nenv)
    last_values : array (nenv,)
    gamma :
    lam :
    '''
    num_steps = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    last_gae_lam = 0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            # last timestep, will be executed first
            next_values = last_values
        else:
            # all other timesteps
            next_values = values[t+1]
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values * non_terminal - values[t]
        advantages[t] = last_gae_lam = delta + gamma * lam * non_terminal * last_gae_lam
    return advantages


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
