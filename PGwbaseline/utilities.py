#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 12:52:36 2019

@author: chryskan
"""
import numpy as np
from skimage.color import rgb2gray
from skimage import transform
from collections import deque
from params import *

# =============================================================================
# def preprocess(state):
#     state = rgb2gray(state)
#     state = state[9:-13,4:-15]
#     # for space invaders : [9:-13,4:-15] is better than [8:-12, 4:-12]
#     state = state/255.0
#     state = transform.resize(state, [max_height, max_width])
#     
#     return state
# 
# def stack_frames(stack, state, is_new):
#     state = preprocess(state)
#     if is_new:
#         stack = deque([np.zeros((max_height, max_width), dtype= np.int) \
#                        for i in range(STACK_SIZE)], maxlen=STACK_SIZE)
#         for i in range(1,5):
#             stack.append(state)
#     else:
#         stack.append(state)
#     stacked_state = np.stack(stack, axis=2)
#     return stacked_state, stack
# 
# =============================================================================
def discount_rewards(r):
    discounted_r = np.zeros_like(r, dtype=np.float32)
    total_r = 0
    for t in reversed(range(r.size)):
        total_r = total_r * GAMMA + r[t]
        discounted_r[t] = total_r
    mean = np.mean(discounted_r)
    std = np.std(discounted_r)
    std = std if std != 0 else 1
    discounted_r =  (discounted_r - mean) / std
    return discounted_r
