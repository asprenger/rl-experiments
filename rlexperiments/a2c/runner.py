import numpy as np

class Runner(object):

    def __init__(self, env, num_env, model, valid_actions, num_steps=5, gamma=0.99):
        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space.shape # 84, 84, 1
        self.nc = nc # number of channels in an environment state/observation

        # TODO fix for Mujoco
        self.batch_ob_shape = (num_env * num_steps, 84, 84, 4)

        # Storage for the last state for each environment. This variable
        # also stores the state between invocation of run().
        self.obs = np.zeros((num_env,) + env.observation_space.shape, dtype=model.train_model.X.dtype.name)
        self.obs[:] = env.reset()

        self.gamma = gamma
        self.num_steps = num_steps
        self.num_env = num_env


        self.valid_actions = valid_actions

        self.episode_rewards = np.zeros(num_env)
        self.running_rewards = np.full((num_env,), None)

        self.episode_steps = np.zeros(num_env)
        self.running_steps = np.full((num_env,), None)

        self.running_values = np.full((num_env,), None)


    def run(self):

        mb_obs = []
        mb_rewards = []
        mb_actions = []
        mb_values = []
        mb_dones = []

        for n in range(self.num_steps):

            # sample actions and estimate values for the current state
            actions, values = self.model.step(self.obs)

            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)

            self.obs[:], rewards, dones, _ = self.env.step(self.valid_actions[actions])

            mb_rewards.append(rewards)
            mb_dones.append(dones)



            self.episode_rewards += rewards
            self.episode_steps += np.ones(self.num_env)
            for i, done in enumerate(dones):
                if done:
                    self.running_rewards[i] = self.episode_rewards[i] if self.running_rewards[i] is None else self.running_rewards[i] * 0.99 + self.episode_rewards[i] * 0.01
                    self.episode_rewards[i] = 0
                    self.running_steps[i] = self.episode_steps[i] if self.running_steps[i] is None else self.running_steps[i] * 0.99 + self.episode_steps[i] * 0.01
                    self.episode_steps[i] = 0

            for i, value in enumerate(values):        
                self.running_values[i] = values[i] if self.running_values[i] is None else self.running_values[i] * 0.99 + values[i] * 0.01
            

        # batch of steps to batch of rollouts

        # TODO fix for Mujoco
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape) # (num_env * num_steps, 84, 84, 4)
        
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0) # (num_env, num_steps)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0) # (num_env, num_steps)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0) # (num_env, num_steps)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)


        # discount rewards / bootstrap off value function
        last_values = self.model.value(self.obs).tolist()
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            # n iterate over environments
            # rewards and dones have shape (num_steps,)
            # value is a scalar
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                # if the last step is not an episode end bootstrap rewards discounting
                # from the value estimation of the next state
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards


        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()

        return mb_obs, mb_rewards, mb_actions, mb_values

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1. - done)
        discounted.append(r)
    return discounted[::-1]
