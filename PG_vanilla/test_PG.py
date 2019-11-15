#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 17:16:07 2019

@author: chryskan
"""
import numpy as np
from params import *
import os

cwd = os.getcwd() 
path = cwd+"/models/"+NAME+"/" + GAME
suffix ="_actor_p.h5"
p = os.path.join(path, NAME +suffix)
RENDER = True

def test(Agent, env):
    n_episodes = 50
    test_history = []
    for i in range(n_episodes):
        done = False
        score = 0
        observation = env.reset()
    
        while not done:
            #action_index, action_vector, action_prob = get_action(observation)
            action_vector, action_index, action_prob = Agent.predict_action(observation)
            observation_, reward, done, info = env.step(action_index)
            if RENDER == True:
                env.render()
            observation = observation_
            score += reward
        print(score)
        test_history.append(score)
    print("**********\nTEST\n average score: ", np.mean(test_history),
          "\nvariance: ", np.std(test_history),"\n**********")
    return  np.mean(test_history), np.std(test_history)
#env.close()
