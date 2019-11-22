#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:43:47 2019

@author: chryskan
"""

import tensorflow as tf
import numpy as np
from keras.layers import Conv2D, Dense, Flatten, Input
from keras.layers import MaxPooling2D, Dropout, Concatenate, Softmax, BatchNormalization# to add
from keras.activations import relu
from params import *
#from utilities import *
import pickle as pkl
from keras.optimizers import Adam, RMSprop
from keras.models import Model
import tensorflow.keras.backend as K

class PPONetwork():
    def __init__(self, graph=None, input_shape=INPUT_SHAPE,
                 action_size=ACTION_SPACE_SIZE, lambda_=LAMBDA, horizon=HORIZON,
                 conv_params=CONV_PARAMS, dense_params=DENSE_PARAMS, 
                 gamma=GAMMA, batch_size=BATCH_SIZE, learning_rate = LEARNING_RATE,
                 cr_lr = CR_LR, clip_e=CLIP_E, name=NAME):
        self.input_shape = input_shape
        self.action_size = action_size
        self.gamma = gamma
        self.lr = learning_rate
        self.cr_lr = cr_lr
        self.vf_coef = 1
        self.memory = []
        self.I = 1
        self.name = name
        self.beta = 0.01
        self.clip_e = clip_e
        self.lambda_ = lambda_
        self.horizon = horizon
        self.callbacks = []
        self.actor_t, self.critic, self.actor_p = \
            self._create_ppo_net(conv_params, dense_params)

    def _create_ppo_net(self, conv_params=None, dense_params=[64,64]):
        
        inputs_ = Input(shape=self.input_shape)
        advantage_ = Input(shape=[1])
        reward_ = Input(shape=[1])
        value_ = Input(shape=[1])
        old_policy = Input(shape=[self.action_size])
        out = inputs_

        for units in dense_params:
            out = Dense(units=units, activation='relu')(out)    
        
        policy_ = Dense(units=self.action_size, activation='softmax')(out)
        t_value_ = Dense(units=1, activation='linear')(out)

        policy = Model(inputs=inputs_, outputs=policy_)
        
        def ppo_loss(adv, old_pred, reward, value):    
            def _loss(y_true, y_pred):
                prob = K.mean(y_true * y_pred)
                old_prob = K.mean(y_true * old_pred)
                r = K.exp(K.log(prob+ 1e-10) - K.log(old_prob + 1e-10))
                
                p1 = r*adv
                p2 = K.clip(r, 1-self.clip_e,1+self.clip_e)*adv
                
                ppo_loss = K.minimum(p1, p2)
                entropy = -prob*K.log(prob + 1e-10)
                value_loss = K.mean(K.square(reward - value))
                return -ppo_loss + self.vf_coef*value_loss - self.beta * entropy
            return _loss
        
        def critic_loss(y_true, y_pred):
            return K.mean(K.square(y_true - y_pred))  
     
        actor = Model(inputs=[inputs_, advantage_, 
                              old_policy, reward_, value_], outputs=policy_)
        opt_ac = Adam(lr=self.lr)
        self.actor_opt = opt_ac

        actor.compile(optimizer=opt_ac, 
                      loss=ppo_loss(advantage_, old_policy, 
                                    reward_, value_))
        
        critic = Model(inputs=inputs_, outputs=t_value_)
        opt_cr = Adam(lr=self.cr_lr)
        self.critic_opt = opt_cr     
        critic.compile(optimizer=opt_cr, loss=critic_loss)
  
        return actor, critic, policy
    
    def store_transition(self, state, action, reward,
                         prob_dist, next_state, done):
        t = (state, action, reward, prob_dist, next_state, done)
        self.memory.append(t)
            
    def get_action(self, state):
        state = np.expand_dims(state, axis=0)
        action_prob_dist = self.actor_p.predict(state)[0]
        action_index = np.random.choice(self.action_size, p=action_prob_dist)
        action_vector = ACTION_SPACE[action_index]
        return action_index, action_vector, action_prob_dist

    def GAEs(self):
        # S: States, A : actions, R: rewards, P: probability distributions,
        # N: Next states, D: dones, V: values, M: masks
        S,A,R,P,N,D = [np.array(e) for e in zip(*self.memory)]
        V = self.critic.predict(S)
        if D[-1]==1:
            V = np.append(V,0)
        else:
            V = np.append(V,self.critic.predict(np.expand_dims(N[-1], axis=0)))
        Returns = []
        M = 1-D
        gae = 0
        for i in reversed(range(len(R))):
            delta = R[i]* + self.gamma * V[i+1]*M[i] -V[i]
            gae = delta + self.gamma*self.lambda_*M[i]*gae
            Returns.insert(0, gae + V[i])
        advs = np.array(Returns) - V[:-1]
        return np.array(Returns), (advs - np.mean(advs))/ (np.std(advs)+1e-10)
    

    def train(self):
        '''This is used at the end of an episode'''
        Returns, Advs = self.GAEs()
        Xs, Acts, Rs, Probs, Ns, Dones = [np.array(e) for e in zip(*self.memory)]
        self.memory = []
        Vs = self.critic.predict(Xs)
        history1 = self.actor_t.fit([Xs, Advs, Probs, Returns, Vs], 
                                    y=Acts, epochs=EPOCHS, 
                                    verbose=False, shuffle=True)
        history2 = self.critic.fit(x=Xs, y=Returns, epochs=EPOCHS, 
                                   verbose=False, shuffle=True)
        
        return history1.history['loss'], history2.history['loss']
    
    def learn(self, state, action, reward, state_, done):
        '''This is used after every step'''
        state = state[np.newaxis, :]
        state_ = state_[np.newaxis, :]
        
        critic_value_ = self.critic.predict(state_)
        critic_value = self.critic.predict(state)
         
        target = reward + self.gamma*critic_value_*(1-int(done))
        delta = target - critic_value
        
        action_prob_dist = self.actor_p.predict(state)      
        actions = np.zeros([1, self.action_size])         
        actions[np.arange(1), action] = 1.0
        self.actor_t.fit([state, delta, action_prob_dist, target, critic_value],
                         actions, epochs=3,
                         verbose=0, shuffle=True)
        self.critic.fit(state, target, epochs=3, verbose=0, shuffle=True)
