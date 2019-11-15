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

cwd = os.getcwd() 
path = cwd+"/models/"+NAME+"/space_invaders/"
os.makedirs(path, exist_ok=True)

class PGNetwork():
    def __init__(self, input_shape=INPUT_SHAPE, action_size=ACTION_SPACE_SIZE,
                 conv_params=CONV_PARAMS, dense_params=DENSE_PARAMS, 
                 gamma=GAMMA, batch_size=BATCH_SIZE, learning_rate = LEARNING_RATE,
                 name=NAME):
        self.input_shape = input_shape
        self.action_size = action_size
        self.gamma = gamma
        self.lr = learning_rate
        self.pg_network = self._create_network(conv_params, dense_params)
        self.memory = []
        
        self.name = name
        
    def _create_network(self, conv_params=None, dense_params=[64,64]):
        
        inputs_ = Input(shape=self.input_shape)
        out = inputs_
# =============================================================================
#         for index, (fl, ks, st) in enumerate(conv_params):
#             out = Conv2D(filters=fl, kernel_size=(ks,ks), strides=(st,st))(out)
#           #  out = BatchNormalization(epsilon=1e-5)(out)
#             out = ReLU()(out)
#         out = Flatten()(out)
# =============================================================================
        for units in dense_params:
            out = Dense(units=units, activation='relu')(out)    #logits
        
        outputs_ = Dense(units=self.action_size, activation='softmax')(out)

        model = Model(inputs=inputs_, outputs=outputs_)
        opt = Adam(lr=self.lr)
        model.compile(optimizer=opt, loss='categorical_crossentropy')
        
        return model
    
    def get_action(self, state):
        state = np.expand_dims(state, axis=0)
        action_prob_dist = self.pg_network.predict(state)[0]
        action_index = np.random.choice(self.action_size, p=action_prob_dist)
        action_vector = ACTION_SPACE[action_index]
        return action_index, action_vector
    
    def store_batch(self, S, A, R):
        Y = discount_rewards(np.vstack(R))
        Y = A*np.clip(Y, 0, 1)
        batch = (np.array(S), Y)
        self.memory.append(batch)
        
    def train(self):
        Xs, Ys = zip(*self.memory)
        self.memory = []
        Xs, Ys = np.vstack(Xs), np.vstack(Ys)
        print(Xs.shape, Ys.shape)

        history = self.pg_network.fit(x=Xs, y=Ys, batch_size=BATCH_SIZE, verbose=True)
        return history.history['loss']

def save_model():
    p = os.path.join(path, NAME+"checkpoint.h5")
    Agent.pg_network.save(p)
    print("Model Saved")
    return 
    

state = env.reset()
#state, stack = stack_frames(None, state, True)
Agent = PGNetwork()
S, A, R = [], [], []
counter = 0
episode_reward = 0
all_rewards , all_losses = [], []
TOTAL_TR_STEPS = 100_000
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
       # next_state, stack = stack_frames(stack, next_state, False)
        if done or step == TOTAL_TR_STEPS-1:
            Agent.store_batch(S,A,R)
            S, A, R = [], [], []
            counter += 1
            all_rewards.append(episode_reward)
            episode_reward = 0
            next_state = env.reset()
         #   next_state, stack = stack_frames(None, next_state, True)
            if counter % BATCH_SIZE == 0:
                loss = Agent.train()
                all_losses.append(loss)
                counter = 0
      #          save.model()
        state = next_state

except KeyboardInterrupt:
  #  save_model()
    raise
    
env.close()
#save_model()