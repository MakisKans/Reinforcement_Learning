#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 01:36:48 2019

@author: chryskan
"""

import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten, Input, Lambda
from keras.optimizers import Adam
from keras.models import Model
import keras.backend as K
from params import *
import numpy as np
from utilities import *
import matplotlib.pyplot as plt
from Memory import ReplayMemory, ReplayMemoryPer
import random
import os
from time import time

class Agent():
    def __init__(self, input_shape=INPUT_SHAPE, action_shape=ACTION_SHAPE, 
                 learning_rate=LEARNING_RATE, conv_params=CONV_PARAMS, 
                 dense_params=DENSE_PARAMS, gamma=GAMMA, batch_size=BATCH_SIZE, 
                 minibatch_size=MINIBATCH_SIZE, duel_type=DUEL_TYPE, name=NAME):
        self.input_shape = input_shape
        self.action_shape = action_shape
        self.learning_rate= learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.name = name
        self.memory = ReplayMemoryPer(max_size=MEMORY_SIZE)
        self.epsilon = 1
        self.eps_dec = 1/DECAY_STEPS #1e-5
        self.duel_type = duel_type
        self.q_net = self._build_dqn(conv_params, dense_params)
        self.target_q_net = self._build_dqn(conv_params, dense_params)
        self.update_target()
        
    def _build_dqn(self, conv_params, dense_params):
        inputs_ = Input(shape=self.input_shape)
        out = inputs_
        
        for index, (fl, ks, st) in enumerate(conv_params):
            out = Conv2D(filters=fl, kernel_size=ks, strides=st,
                         activation='relu')(out)   
        out = Flatten()(out)
        
        for units in dense_params:
            out = Dense(units=units, activation='relu')(out)

        value_fc = Dense(units=VALUE_FC, activation='relu')(out)
        value = Dense(units=1, activation='linear')(value_fc)
        
        adv_fc = Dense(units=ADVANTAGE_FC, activation='relu')(out)
        advantages = Dense(units=self.action_shape, activation='linear')(adv_fc)
        
        if self.duel_type=='ave':
            compute = Lambda(lambda x: x[0] - K.mean(x[0]) + x[1])
        elif self.duel_type=='max':
            compute = Lambda(lambda x: x[0] - K.max(x[0]) + x[1])
        elif self.duel_type=='naive':
            compute = Lambda(lambda x: x[0] + x[1])
        else:
            raise Exception("unknown type for Q(s,a) calculation, choose either ave/max/naive")
        
        outputs = compute([advantages, value])
        model = Model(inputs=inputs_, outputs=outputs)
        opt = Adam(lr=self.learning_rate, epsilon=ADAM_EPS)
        model.compile(optimizer=opt, loss='mean_squared_error')
        
        return model
    
    def _path(self):
        cwd = os.getcwd() 
        path = cwd+"/models/"+GAME+"/"
        os.makedirs(path, exist_ok=True)
        return path
    
    def save(self):
        p = self._path() + self.name +".ckpt"
        self.q_net.save_weights(p)

    def load(self):
        p = self._path() + self.name +".ckpt"
        self.q_net.load_weights(p)
        
    def predict_action(self, state):
        if self.epsilon > np.random.rand():
            action = random.randint(0, len(ACTIONS_SPACE) - 1)
        else:
            state = np.expand_dims(state.__array__(), axis=0)
            Qs = self.q_net.predict(state)
            action = np.argmax(Qs)
        self.epsilon = self.epsilon - (self.eps_dec) \
            if self.epsilon > EPS_MIN else EPS_MIN
        return action
    
    def bootstrap(self, rewards, next_states, dones):
        if DOUBLE:
            Qs = self.target_q_net.predict(next_states)
            Qs_ = self.q_net.predict(next_states)
            actions = np.argmax(Qs_, axis=-1)
            rows = np.arange(Qs.shape[0])
            return rewards + self.gamma * Qs[rows,actions]*(1-dones)
        else:
            Qs = self.target_q_net.predict(next_states)
            actions = np.argmax(Qs, axis=-1)
            rows = np.arange(Qs.shape[0])
            return rewards + self.gamma * Qs[rows,actions]*(1-dones)
        
    def update_target(self):
        self.target_q_net.set_weights(self.q_net.get_weights())
    
    def _turn_to_array(self, experience_sequence):
        states, actions, rewards, next_states, dones = zip(*experience_sequence)
        states = [s.__array__() for s in states]
        next_states = [s.__array__() for s in next_states]
        return [np.array(e) for e in [states, actions, rewards, next_states, dones]]
    
    def train(self, step):
        if self.memory.size < MIN_MEMORY:
            return None
        
        indices, transitions, IS_weights = self.memory.sample(step)
        index_range = np.arange(transitions.shape[0])
        exp_seq = transitions[index_range,2]
        states, actions, rewards, next_states, dones = self._turn_to_array(exp_seq)
     #   states, actions, rewards, next_states, dones =\
        #        self.memory.sample(self.minibatch_size)
                
        p_Qs = self.q_net.predict(states)
        t_Qs = np.copy(p_Qs)
       # index_range = np.arange(states.shape[0])
        t_Qs[index_range, actions] = self.bootstrap(rewards, next_states, dones)
# =============================================================================
#         history = self.q_net.fit(states, t_Qs,
#                                  batch_size=self.batch_size,                                     
#                                  verbose=False,shuffle=False)
# =============================================================================
        errors = np.sum(np.abs(t_Qs - p_Qs), axis=1)

        history = self.q_net.fit(states, t_Qs,
                                     batch_size=self.batch_size,
                                     sample_weight=IS_weights,
                                     verbose=False,
                                     shuffle=False)
        
        self.memory.batch_update(indices, errors)
        
        return history.history["loss"]

def initial_observing(agent, env):
    state = env.reset()
    print("Populating experience replay buffer...")
    for i in range(INITIAL_OBS):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        experience = (state, action, reward, next_state, done)
        agent.memory.append(experience)
        state = next_state
        if done:
            state = env.reset()
    env.reset()

