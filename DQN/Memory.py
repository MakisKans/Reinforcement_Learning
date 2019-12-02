#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 12:41:16 2019

@author: chryskan
"""

import random
import numpy as np
from utilities import update_stack
from params import STACK_SIZE, MINIBATCH_SIZE, MIN_MEMORY

#MINIBATCH_SIZE = 64

class ReplayMemory():
    def __init__(self, max_size=10000):
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def append(self, obj):
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size=MINIBATCH_SIZE):
        if self.size < MIN_MEMORY:
            return None
        if self.size - 2*(STACK_SIZE-1) < batch_size:
            batch_size = self.size- 2*(STACK_SIZE-1)
        indices = random.sample(range(STACK_SIZE-1, 
                                      self.size-STACK_SIZE+1),
                                batch_size)
        return [self._get_and_stack(index) for index in indices]
    
    def _get_and_stack(self, index):
        '''this hidden function gets 4 consecutive 
        frames from memory and stacks them'''
        #(s,a,r,s_,d) = self.buffer[index]
        #return self.buffer[index]
        ###############################
        start = 0
        for i in range(1,STACK_SIZE):
            (s,a,r,s_,d) = self.buffer[index - i]
            if d:
                break
            start = i
            
        (s,a,r,s_,d) = self.buffer[index - start]
        
        state = update_stack(None, s, True)
        next_state = update_stack(None, s, True)
        next_state = update_stack(next_state, s_)
        
        for i in range(index-start+1, index+1):
            (s,a,r,s_,d) = self.buffer[i]
            state = update_stack(state, s)
            next_state = update_stack(next_state, s_)
        return state, a, r, next_state, d
        ###############################