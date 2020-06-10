#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 12:41:16 2019

@author: chryskan
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 16:15:56 2019

@author: chryskan
"""
import random
import numpy as np
from params import *
#STACK_SIZE, MINIBATCH_SIZE, MIN_MEMORY

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
        exps = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*exps)
        states = [s.__array__() for s in states]
        next_states = [s.__array__() for s in next_states]
        return [np.array(e) for e in [states, actions, rewards, next_states, dones]]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 12:41:16 2019

@author: chryskan
"""
from SumTree import SumTree

# prioritized replay with sum tree implementation

class ReplayMemoryPer():
    def __init__(self, max_size=10000, minibatch_size=MINIBATCH_SIZE, alpha = ALPHA_PER, beta = BETA_PER, eps=EPS_PER, steps=TOTAL_TR_STEPS):
        self.size = max_size
        self.sumTree = SumTree(capacity=max_size)
        self.minibatch_size = minibatch_size
        self.alpha = alpha
        self.beta = beta
        self.beta_cur = beta
        self.steps = steps
        self.eps = eps
        
    def append(self, obj):
       # if max_p == 0:
       #     max_p = self.abs_err_upper
        self.sumTree.add(self.sumTree.max_p, obj)
    
    def _beta_annealing(self, step):
        return step*(1-self.beta)/self.steps + self.beta

    def _comp_weight(self, P):
        N = self.sumTree.capacity
        return (N*P)**(-self.beta_cur)
    
    def batch_update(self, tree_idxs, errors):
        errors = [e+self.eps for e in errors]
        # TODO
        #clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        for ti, e in zip(tree_idxs, errors):
            p = e**self.alpha
            self.sumTree.update(ti, p)

    def sample(self, step):
        #if size of memory less than minibatch_size
        batch_size = self.minibatch_size
        
        total = self.sumTree.total()
        delta = total/batch_size
        self.beta_cur = self._beta_annealing(step)

        priorities = [np.random.rand()*delta + delta*(k-1) for k in range(1, batch_size+1)]

        transitions = np.asarray([self.sumTree.get(p) for p in priorities]) 
        # index, priority, data
        indices = [t[0] for t in transitions]
        probs = [t[1]/self.sumTree.total() for t in transitions]
       
        IS_weights = [self._comp_weight(P) for P in probs]
        max_w = np.max(IS_weights)
        IS_weights = np.asarray([w/max_w for w in IS_weights])
        
        return indices, transitions, IS_weights