import numpy as np
import time
from random import randint
import gym

def main():

    env_id = 'Reacher-v2'
    seed = randint(0, 1000000)
    
    env = gym.make(env_id)
    env.reset()
    
    t = 1
    while True:
        env.render()
        # positive: counter-clock direction
        # negative: clock direction
        # [center_hub, outer_hub]
        # movement of the output joint is limited by the center_hub
        obs, reward, done, info = env.step(np.array([0.0, 0.0]))
        #print('%d: obs=%s reward=%f done=%s info=%s' % (t, obs, reward, done, info))
        print(obs)
        time.sleep(0.1)
        t += 1
        if done:
            print('DONE')
            obs = env.reset()
            t = 1
            time.sleep(2)

if __name__ == "__main__":
    main()
