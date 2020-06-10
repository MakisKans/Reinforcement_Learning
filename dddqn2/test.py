#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 05:26:58 2020

@author: chryskan
"""

from params import env
from dddqn_per import Agent

agent = Agent()
agent.load()
#%%
done = False
state = env.reset()
agent.train_noise = False
#%%
total = 0
while not done:
    action = agent.predict_action(state)
    state, rew, done, _ = env.step(action)
    env.render()
    total+=rew
    if done:
        state = env.reset()
        print(total)
        total = 0
        done = False