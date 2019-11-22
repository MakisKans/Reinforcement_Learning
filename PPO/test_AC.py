#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 17:16:07 2019

@author: chryskan
"""
import numpy as np
from params import *
import os
import tensorflow as tf
from utilities import *

cwd = os.getcwd() 
path = cwd+"/models/"+NAME+"/" + GAME
suffix ="_actor_p.h5"
p = os.path.join(path, NAME +suffix)
RENDER = False

def test(Agent, env):
    n_episodes = 50
    test_history = []
    for i in range(n_episodes):
        done = False
        score = 0
        observation = env.reset()
#        observation, stack = stack_frames(None, observation, True)

        while not done:
            action_index, action_vector, action_prob = Agent.get_action(observation)
            observation_, reward, done, info = env.step(action_index)
            if RENDER == True:
                env.render()
            observation = observation_
          #  observation, stack = stack_frames(stack, observation, False)
            score += reward
        test_history.append(score)
    print("**********\nTEST\naverage score: ", np.mean(test_history),
          "\nstd deviation: ", np.std(test_history),"\n**********")
    env.reset()
    return  np.mean(test_history), np.std(test_history)
#env.close()