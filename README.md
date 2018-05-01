
TensorFlow implementations of different reinforcement learning algorithms.

List of implemented algorithms:

- [Vanilla PG](rlexperiments/pg)
- [A2C](rlexperiments/a2c)
- [DQN](rlexperiments/dqn)
- [PPO](rlexperiments/ppo)

The package can be install by typing:

```bash
git clone https://github.com/asprenger/rl-experiments.git
cd rl-experiments
pip install -e .
```

# Demos

Here is an example of the Roboschool Hopper-v1 trained with PPO. It has learned that it can create more forward momentum by moving
the upper limb forward at the right moment.

<a href="http://www.youtube.com/watch?v=SjIGbM5uqCY"><img src="images/RoboschoolHopper.png" width="360" height="270" target="_blank"/></a>

Here is another example the the Roboschool Walker2d-v1 also trained on PPO. The walking is pretty clumsy and the agent falls over from time to time.

<a href="http://www.youtube.com/watch?v=qmfJQRleo5A"><img src="images/RoboschoolWalker2d.png" width="360" height="270" target="_blank"/></a>
