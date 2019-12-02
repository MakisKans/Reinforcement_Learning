#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 17:04:46 2019

@author: chryskan
"""

import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten, Input, Conv3D
from keras.optimizers import Adam
from keras.models import Model
import keras.backend as K
from params import *
import numpy as np
from utilities import *
import matplotlib.pyplot as plt
from Memory import ReplayMemory
import random
import os

class Agent():
    def __init__(self, input_shape=INPUT_SHAPE, action_shape=ACTION_SHAPE, learning_rate=LEARNING_RATE, conv_params=CONV_PARAMS, dense_params=DENSE_PARAMS, gamma=GAMMA, batch_size=BATCH_SIZE, minibatch_size=MINIBATCH_SIZE, name=NAME):
        self.input_shape = input_shape
        self.action_shape = action_shape
        self.learning_rate= learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.name = name
        self.memory = ReplayMemory(max_size=MEMORY_SIZE)
        self.q_net = self._build_dqn(conv_params, dense_params)
        self.target_q_net = self._build_dqn(conv_params, dense_params)
        self.update_target()
        
    def _build_dqn(self, conv_params, dense_params):
        inputs_ = Input(shape=self.input_shape)
        out = inputs_
# =============================================================================
#         for index, (fl, ks, st) in enumerate(conv_params):
#             out = Conv2D(filters=fl, kernel_size=ks, strides=st,
#                          activation='relu')(out)
#             
#         out = Flatten()(out)
# =============================================================================
        
        for units in dense_params:
            out = Dense(units=units, activation='relu')(out)
        
        outputs_ = Dense(self.action_shape, activation="linear")(out)
        
        model = Model(inputs=inputs_, outputs=outputs_)
        opt = Adam(lr=self.learning_rate, epsilon=ADAM_EPS)
        model.compile(loss="mse", optimizer=opt)
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
        pass
    
    def lin_decay(self, step, explore_start=E_START, explore_stop=E_STOP):
        if step > DECAY_STEPS:
            return EPS_MIN            
        a = explore_start
        b = explore_stop
        return ((b-a)/DECAY_STEPS)*step + a
    
    def predict_action(self, step, state, explore_start=E_START, 
                           explore_stop=E_STOP):
        expl_proba = self.lin_decay(step, explore_start, explore_stop)
        if expl_proba > np.random.rand():
            choice = random.randint(0, len(ACTIONS_SPACE) - 1)
            action = ACTIONS_SPACE[choice]
        else:
            state = np.expand_dims(state, axis=0)
            Qs = self.q_net.predict(state)
            choice = np.argmax(Qs)
            action = ACTIONS_SPACE[choice]
        return action, choice
    
    def bootstrap(self, reward,state):
        state = np.expand_dims(state, axis=0)
        Qs = self.target_q_net.predict(state)
        return reward + self.gamma * np.max(Qs)
    
    def update_target(self):
        self.target_q_net.set_weights(self.q_net.get_weights())
   
    def train(self):
        if self.memory.size < MIN_MEMORY:
            return None
        exp_seq = self.memory.sample(self.minibatch_size)
        curr_states = []
        batch_target_Qs = []

        for exp in exp_seq:
            (state, action, reward, next_state, done) = exp
            curr_states.append(state)
            pred_Qs = self.target_q_net.predict(np.expand_dims(state, axis=0))
            if not done:
                target_Q = self.bootstrap(reward,next_state)
            else:
                target_Q = reward
            target_Qs = pred_Qs * (np.ones(self.action_shape) - action)\
                        + target_Q * action
            batch_target_Qs.append(np.squeeze(target_Qs))
            
        curr_states = np.array(curr_states)
        batch_target_Qs = np.array(batch_target_Qs)

        history = self.q_net.fit(curr_states, batch_target_Qs,
                                 batch_size=self.batch_size,                                     
                                 verbose=False,shuffle=False)
        return history.history["loss"]

def initial_observing(agent, env):
    with tf.Session() as sess:    
        obs = env.reset()
        #state = obs ##!!
        ###############################
        state = image_transformer.transform(obs, sess)
        ###############################
        print("Populating experience replay buffer...")
        for i in range(INITIAL_OBS):
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            ###############################
            next_state = image_transformer.transform(obs, sess) 
            ###############################
            #next_state = obs ##!!
            experience = (state, action, reward, next_state, done)
            agent.memory.append(experience)
            state = next_state
            if done:
                obs = env.reset()
                #state = obs ##!!
                ###############################
                state = image_transformer.transform(obs, sess)
                ###############################
    env.reset()

def random_noops(env):
    obs = env.reset()
    noops = np.random.randint(NOOP+1)
    for i in range(noops):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        if done:
            obs = env.reset()
    return obs
#%%
agent = Agent()
image_transformer = ImageTransformer()
initial_observing(agent, env)
#%%
sess = tf.Session()
episode_reward = 0
score_history = []
episode = 0
###############################
obs = random_noops(env)
obs = image_transformer.transform(obs, sess)
state = update_stack(None, obs, True)
###############################
#obs = env.reset() ##!!
#state = obs ##!!
#%%
try:
    for t in range(1, TOTAL_TR_STEPS+1):
        action, choice = agent.predict_action(step=t, state=state)
        
        for i in range(K_):
            obs_, reward , done , info = env.step(choice)
            episode_reward += reward
            ###############################
            obs_ = image_transformer.transform(obs_, sess)
            ###############################
            experience = (obs, action, reward, obs_, done)
            agent.memory.append(experience)
            obs = obs_
            ###############################
            state = update_stack(state, obs)
            ###############################
            #state = obs ##!!
            if done:
                break
        
        
        if t % TRAIN_EVERY == 0:
            loss = agent.train()
        if t % TARGET_UPDATE == 0 :
            agent.update_target()            
        if t % TEST_EVERY == 0 and t>0:
            _,_ = test_agent(agent, image_transformer, sess)
            
        if done \
        or t==TOTAL_TR_STEPS-1\
        or (t % MAX_STEPS == 0 and t>0):
            episode+=1
            score_history.append(episode_reward)
            if episode % 10 == 0:
                avg_score = np.mean(score_history[-100:])
                print("At step: ", t)
                print('episode ', episode, 'score %.2f average score %2f' % \
                  (episode_reward, avg_score))
            episode_reward = 0
            ###############################
            obs = random_noops(env)
            obs = image_transformer.transform(obs, sess)
            state = update_stack(None, obs, True)
            ###############################
            #obs = env.reset() ##!!
            #state = obs ##!!
        
except KeyboardInterrupt:
    agent.save()
    raise

agent.save()
env.close()