#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 21:07:17 2019

@author: chryskan
"""

import numpy as np
from keras.layers import Conv2D, Dense, Flatten, Input, ReLU
from keras.layers import MaxPooling2D, Dropout, Concatenate, Softmax, BatchNormalization# to add
from keras.activations import relu
from params import *
from utilities import *
import pickle as pkl
from keras.optimizers import Adam, RMSprop
from keras.models import Model
import keras.backend as K
import os
from test_PG import test
from time import time


class PGNetwork():
    def __init__(self, input_shape=INPUT_SHAPE, action_size=ACTION_SPACE_SIZE,
                 conv_params=CONV_PARAMS, dense_params=DENSE_PARAMS, 
                 gamma=GAMMA, learning_rate = LEARNING_RATE, name=NAME):
        self.input_shape = input_shape
        self.action_size = action_size
        self.gamma = gamma
        self.lr = learning_rate
        self.pg_net = self._create_policy_net(conv_params, dense_params)
        self.v_net = self._create_value_net(conv_params, dense_params)
        self.memory = []
        
        self.name = name
        
    def _create_policy_net(self, conv_params=None, dense_params=[64,64]):
        
        inputs_ = Input(shape=self.input_shape)
        out = inputs_
        for units in dense_params:
            out = Dense(units=units, activation='relu')(out)    #logits
        
        outputs_ = Dense(units=self.action_size, activation='softmax')(out)

        model = Model(inputs=inputs_, outputs=outputs_)
        opt = Adam(lr=self.lr)
        model.compile(optimizer=opt, loss='categorical_crossentropy')
        
        return model
    
    def _create_value_net(self, conv_params=None, dense_params=[64,64]):
        inputs_ = Input(shape=self.input_shape)
        out = inputs_

        for units in dense_params:
            out = Dense(units=units, activation='relu')(out)    #logits
        
        outputs_ = Dense(units=1, activation='linear')(out)

        model = Model(inputs=inputs_, outputs=outputs_)
        opt = Adam(lr=self.lr)
        model.compile(optimizer=opt, loss='mean_squared_error')
        
        return model
    
    def get_action(self, state):
        state = np.expand_dims(state, axis=0)
        action_prob_dist = self.pg_net.predict(state)[0]
        action_index = np.random.choice(self.action_size, p=action_prob_dist)
        action_vector = ACTION_SPACE[action_index]
        return action_index, action_vector
    
    def store_batch(self, S, A, R):
        Y = np.clip(discount_rewards(np.vstack(R)), 0, 1)
        batch = (np.array(S), A, Y)
        self.memory.append(batch)
        
    def train(self):
        Xs, Acts, Gs = zip(*self.memory)
        self.memory = []
        Xs, Acts, Gs = np.vstack(Xs), np.vstack(Acts), np.vstack(Gs)
        Vs = self.v_net.predict(Xs)
        As = (Gs - Vs)*Acts
        history1 = self.pg_net.fit(x=Xs, y=As, verbose=False)
        history2 = self.v_net.fit(x=Xs, y=Gs, verbose=False)
        
        return history1.history['loss'], history2.history['loss']

def save_model():
    cwd = os.getcwd() 
    path = cwd+"/models/"+NAME+"/"+GAME
    os.makedirs(path, exist_ok=True)
    p = os.path.join(path, NAME+"checkpoint.h5")
    Agent.pg_net.save(p)
    print("Model Saved")
    return 

state = env.reset()
Agent = PGNetwork()
S, A, R = [], [], []
counter = 0
episode_reward = 0
score_history = []
test_score_hist, test_std_hist = [], []
episode = 0
cofep = 0
start = time()
print(TOTAL_TR_STEPS)
print(UPDATE)
try:
    for step in range(TOTAL_TR_STEPS):
        if step % 1000 == 0:
     #       save_model()
            print("Step no: " + str(step))
        
        action_index, action_vector = Agent.get_action(state)
        next_state, reward, done, info = env.step(action_index)
        S.append(state)
        A.append(action_vector)
        R.append(reward)
        episode_reward += reward
        if done or step == TOTAL_TR_STEPS-1:
            Agent.store_batch(S,A,R)
            S, A, R = [], [], []
            counter += 1
            score_history.append(episode_reward)
            next_state = env.reset()
            avg_score = np.mean(score_history[-100:])
            print(f"At episode: {counter}")
            print('tr score %.2f average score %2f' % (episode_reward, avg_score))
            episode_reward = 0
            if counter % UPDATE == 0:
                Mean, Var = test(Agent,  env)
                test_score_hist.append(Mean)
                test_std_hist.append(Var)
                _ = Agent.train()

except KeyboardInterrupt:
    #save_model()
    raise
    
env.close()
#save_model()