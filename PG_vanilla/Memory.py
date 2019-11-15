#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 12:41:16 2019

@author: chryskan
"""

import random

MINIBATCH_SIZE = 64

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
        if self.size < batch_size:
            batch_size = self.size
        indices = random.sample(range(self.size), batch_size)
        return [self.buffer[index] for index in indices]