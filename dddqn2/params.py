#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 17:05:48 2019

@author: chryskan
"""
import gym
import numpy as np
from atari_wrappers import make_atari, wrap_deepmind

GAME = 'PongNoFrameskip-v4'
env = wrap_deepmind(make_atari(GAME), episode_life=True, clip_rewards=True, 
                  frame_stack=True, scale=True)
#%% Params
height, width = 84, 84
STACK_SIZE = 4
#INPUT_SHAPE = [height, width, STACK_SIZE]
INPUT_SHAPE = env.observation_space.shape ##!!
ACTION_SHAPE = env.action_space.n

LEARNING_RATE = 0.00025/4
ALPHA_DECAY = 0.01
ADAM_EPS = 1.5e-4
CONV_PARAMS = [(32,8,4),(64,4,2),(64,3,1)] #[(16, 8, 4), (32, 4, 2)] 
DENSE_PARAMS = [512]  #[256]  

GAMMA = 0.99
### Training Hyperparameters

MAX_STEPS = 108000   # Max possible steps in an episode #HORIZON = 128
TOTAL_TR_STEPS = 200000000
DECAY_STEPS = 4000000

MINIBATCH_SIZE = 32
BATCH_SIZE = 32
MEMORY_SIZE = 100000
MIN_MEMORY = 1000
STACK_SIZE = 4

RENDER = False
ACTIONS_SPACE = np.array(np.identity(ACTION_SHAPE, dtype=int).tolist())

TARGET_UPDATE = 32000 #10_000
TRAIN_EVERY = 4
K_ = 4
TEST_EVERY = 1000000
TEST_STEPS = 50000

NOOP = 30
INITIAL_OBS = 80000
E_START = 1
E_STOP = 0.01
EPS_MIN = 0.01

VALUE_FC = 512
ADVANTAGE_FC = 512
ALPHA_PER = 0.7
BETA_PER = 0.5
EPS_PER = 0.01 # minimum ammount to avoid transitions from having 0 priority

DUEL_TYPE = "ave" # or "max"
DOUBLE = True
NAME = "DQN"
