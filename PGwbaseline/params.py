#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 23:50:57 2019

@author: chryskan
"""
import gym
import numpy as np

GAME = "CartPole-v0"
env = gym.make(GAME)
INPUT_SHAPE = [*env.observation_space.shape]
ACTION_SPACE_SIZE = env.action_space.n
ACTION_SPACE = np.array(np.identity(ACTION_SPACE_SIZE, dtype=int).tolist())

LEARNING_RATE = 0.0005 # 0.00025 or 0.000001
GAMMA = 0.99
TOTAL_TR_STEPS = 100_000
UPDATE = 4

RENDER = False

CONV_PARAMS =  None #[(32, 8, 4), (64, 4, 2), (64, 3, 1)]
DENSE_PARAMS = [512]

NAME = "PGwbase"
