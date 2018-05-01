from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))

setup(name='rlexperiments',
      packages=[package for package in find_packages()
                if package.startswith('rlexperiments')],
      install_requires=[
          'gym[mujoco,atari]',
          'tensorflow>=1.4.1'
      ],
      description='RL-experiments: implementations of reinforcement learning algorithms',
      author='Andre Sprenger',
      url='https://github.com/asprenger/rl-experiments',
      version='0.0.1')





