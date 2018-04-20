import time
from random import randint
from rlexperiments.common.atari_utils import make_atari, wrap_deepmind

def main():

    env_id = 'PongNoFrameskip-v4'
    seed = randint(0, 1000000)
    
    env = make_atari(env_id, frame_skip=1)
    env.seed(seed)
    env = wrap_deepmind(env)
    env.reset()
    
    t = 1
    while True:
        env.render()
        next_obs, reward, done, info = env.step(env.action_space.sample())
        print('%d: reward=%f done=%s info=%s' % (t, reward, done, info))

        if done:
            print('DONE')
            next_obs = env.reset()
            t = 1
            time.sleep(2)

if __name__ == "__main__":
    main()
