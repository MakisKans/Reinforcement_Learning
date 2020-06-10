#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 18:11:07 2019

@author: chryskan
"""

import numpy as np

class SumTree:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity -1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.max_p = 1
        
    def _propagate(self, idx, change):
        ''' if a change happens at a leaf, the same change 
        must happen at every parent node, which exists at 
        half the index. Remember that root is idx : 0 hence idx-1//2 '''
        if idx <1:
            print("WTF")
            print(idx, self.max_p, change)
            return
        parent = (idx - 1)// 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        ''' starting from idx, search for s. 
        Every first call will be with idx 0'''
        left = 2*idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])
        
    def total(self):
        return self.tree[0]
    
    def add(self, p, data):
        ''' First call adds at the first node at idx, capacity -1 and continues from there, p : priority'''
        idx = self.write + self.capacity -1
        self.data[self.write] = data
        self.update(idx, p)
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0 # wrap cyclically if exceed capactiy of data
        
    def update(self, idx, p):
        '''p : new priority value'''
        change = p - self.tree[idx]
        self.max_p = max(self.max_p, p)
        self.tree[idx] = p
        self._propagate(idx, change)
    
    def get(self, s):
       # print("Getting: ",s,"from the sumtree")
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        # returns index inn the tree, priorioty , object
        return (idx, self.tree[idx], self.data[dataIdx])