#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 23:50:57 2019

@author: chryskan
"""
import gym
import numpy as np

STACK_SIZE = 4
max_width = 84
max_height = 84
INPUT_SHAPE = [max_height, max_width, STACK_SIZE]

#env = gym.make("SpaceInvaders-v0")
GAME = "CartPole-v0"
env = gym.make(GAME)
INPUT_SHAPE = [*env.observation_space.shape]
ACTION_SPACE_SIZE = env.action_space.n
ACTION_SPACE = np.array(np.identity(ACTION_SPACE_SIZE, dtype=int).tolist())

LEARNING_RATE = 0.0005 #0.00015 # 0.00025 or 0.000001
GAMMA = 0.99
INITIAL_NO_OP = 30

UPDATE_TARGET_STEPS = 10000
TOTAL_TR_STEPS = 20000000

MEMORY_SIZE = 100000
BATCH_SIZE = 5
UPDATE_FREQ = 4

MIN_REPLAY_MEMORY_SIZE = 1000
RENDER = False

CONV_PARAMS =  [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
DENSE_PARAMS = [512]

NAME = "Vanilla_PG"
