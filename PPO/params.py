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

LEARNING_RATE = 0.00001
CR_LR = 0.00005
BETA = 0.01
GAMMA = 0.99

TOTAL_TR_STEPS = 100000
CLIP_E = 0.2
EPOCHS = 3

RENDER = False

CONV_PARAMS =  [(16,8,4),(32,4,2)] 
DENSE_PARAMS = [256]
LAMBDA = 0.95
#HORIZON = 128

NAME = "PPO"