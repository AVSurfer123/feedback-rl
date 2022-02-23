#!/usr/bin/env python3

from setuptools import find_packages, setup

setup(name='feedback_rl',
      version='0.1',
      description='Feedback linearization with RL',
      packages=find_packages(),
      python_requires=">=3.6",
      install_requires=[
          'numpy',
          'matplotlib',
          'torch',
          'gym',
          'gym-cartpole-swingup',
          'scipy',
      ]
)
