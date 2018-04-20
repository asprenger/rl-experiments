import time
from random import randint
import gym

def main():

    
    env_id = 'Hopper-v2'
    seed = randint(0, 1000000)
    
    env = gym.make(env_id)
    env.reset()
    
    t = 1
    while True:
        env.render()
        next_obs, reward, done, info = env.step(env.action_space.sample())
        print('%d: reward=%f done=%s info=%s' % (t, reward, done, info))
        time.sleep(0.1)
        if done:
            print('DONE')
            next_obs = env.reset()
            t = 1
            time.sleep(2)

if __name__ == "__main__":
    main()
