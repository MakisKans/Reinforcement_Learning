#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:50:31 2019

@author: chryskan
"""


import numpy as np
import tensorflow as tf
from Memory import ReplayMemory
from utilities import *
from keras.models import Model
from keras.layers import Conv2D, Flatten, Dense, Input,Lambda, Softmax, BatchNormalization
from keras.layers import MaxPooling2D, Dropout, Concatenate
from keras.activations import relu
from keras.optimizers import rmsprop, Adam
import keras.backend as K
from params import *
import pickle as pkl

class PGNetwork():
    def __init__(self, input_shape=INPUT_SHAPE, action_size=ACTION_SPACE_SIZE,
                 memory_size=MEMORY_SIZE, conv_params=CONV_PARAMS,
                 dense_params=DENSE_PARAMS, 
                 gamma=GAMMA, batch_size=BATCH_SIZE, learning_rate = LEARNING_RATE,
                 name=NAME):
        self.state_shape = input_shape
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma 
        self.model = self._create_network(conv_params, dense_params)
        self.state_memory , self.action_memory, self.reward_memory = [], [], []
        self.name = name

        return
    
    def _create_network(self, conv_params, dense_params):

        inputs_ = Input(shape=self.state_shape)
        advantages = Input(shape=[1])
        out = inputs_
# =============================================================================
#         for index, (fl, ks, st) in enumerate(conv_params):
#             out = Conv2D(filters=fl, kernel_size=(ks,ks), strides=(st,st))(out)
#             out = BatchNormalization(epsilon=1e-5)(out)
#             out = relu()(out)
# =============================================================================
      #  out = Flatten()(out)
        
        for units in dense_params:
            out = Dense(units=units, activation='relu')(out)    #logits
        
        outputs_ = Dense(self.action_size, activation="softmax")(out)

        model = Model(inputs=[inputs_, advantages], outputs=outputs_)
        opt = rmsprop(lr=self.learning_rate)

        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1- 1e-8)
            log_lik = y_true * K.log(out)   # element-wise multiplication, y_true is one hot
                        
            return K.sum(-log_lik*advantages)   # reduce the one hot row to the only element that matters
            
        model.compile(optimizer=opt, loss=custom_loss)
        
        return model
    
    def run_episode(self, env):
        # as is : one episode : one trajectory
        state = env.reset()
     #   stack = deque([np.zeros((84, 84), dtype= np.int) for i in range(STACK_SIZE)], maxlen= 4)
      #  state, stack = stack_frames(None, state, True)
        episode_reward = 0
        done = False        
       
        while not done:
            action_vector, action_index, action_prob = self.predict_action(state)
            next_state, reward, done, _ = env.step(action_index)
            episode_reward += reward
            
            if RENDER==True:
                env.render()
            
       #     next_state, stack = stack_frames(stack, next_state, False)
            
            self.store_transtion(state, action_vector, reward)
            
            state = next_state
        return episode_reward
    
    def train(self):
        # this trains for one episode
        xs = np.array(self.state_memory)
        actions = np.array(self.action_memory)
        rewards = np.array(self.reward_memory)      
        advantages = discount_rewards(rewards)

        cost = self.model.train_on_batch([xs,advantages], actions)
        
        self.flush()

        return cost
    
    def predict_action(self, state):
        state = np.reshape(state, (1,*state.shape))
        action_prob = self.model.predict([state, np.arange(1)]) [0]    # action probability distribution π(α|s)
        action_index = np.random.choice(ACTION_SPACE_SIZE, p=action_prob)
        action_vector = ACTION_SPACE[action_index]
        return action_vector, action_index, action_prob

    def store_transtion(self, state, action, reward):
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
    
    def flush(self):
        self.state_memory, self.action_memory, self.reward_memory = [], [], []