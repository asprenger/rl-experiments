# Proximal Policy Optimization Algorithm

An implementation of the PPO paper: https://arxiv.org/abs/1707.06347

Run the algorithm on an Atari game:

    python -u -m rlexperiments.ppo.train_atari --env PongNoFrameskip-v4

Run the algorithm on a Mujoco or Roboschool environment:

    python -u -m rlexperiments.ppo.train_mujoco --env Hopper-v2

See help (`-h`) for more options.