#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 17:05:48 2019

@author: chryskan
"""
import gym
import numpy as np

GAME = "Breakout-v0" #"Pong-v0"
env = gym.make(GAME)
#%% Params
height, width = 84, 84
STACK_SIZE = 4
INPUT_SHAPE = [height, width, STACK_SIZE]
#INPUT_SHAPE = env.observation_space.shape ##!!
ACTION_SHAPE = env.action_space.n

LEARNING_RATE = 0.00025/4 #0.0001
ALPHA_DECAY = 0.01
ADAM_EPS = 1.5e-4
CONV_PARAMS = [(32,8,4),(64,4,2),(64,3,1)] #[(16, 8, 4), (32, 4, 2)] 
DENSE_PARAMS = [512]  #[256] [256,256] 

GAMMA = 0.99
### Training Hyperparameters

MAX_STEPS = 108_000   # Max possible steps in an episode #HORIZON = 128
TOTAL_TR_STEPS = 10_000_000 #100_000 #200_000_000
DECAY_STEPS = TOTAL_TR_STEPS/10 #4_000_000

MINIBATCH_SIZE = 32
BATCH_SIZE = 32
MEMORY_SIZE = 1000_000 #100_000#
MIN_MEMORY = 1000
STACK_SIZE = 4

RENDER = False
ACTIONS_SPACE = np.array(np.identity(ACTION_SHAPE, dtype=int).tolist())

TARGET_UPDATE = 32_000 #10_000
TRAIN_EVERY = 4#1_000
K_ = 4
TEST_EVERY = 1_000_000
TEST_STEPS = 500_000

NOOP = 30
INITIAL_OBS = 80_000
E_START = 1
E_STOP = 0.01

NAME = "DQN"