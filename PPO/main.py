#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 08:44:35 2019

@author: chryskan
"""
from PPO2 import *
from time import sleep, time
import pickle as pkl
import tensorflow.keras.backend as K
from test_AC import test
import tensorflow as tf
import os

cwd = os.getcwd() 
path = cwd+"/models/"+NAME+"/" + GAME
os.makedirs(path, exist_ok=True)

def save_model():
    p = os.path.join(path, NAME)
    tf.keras.models.save_model(Agent.actor_p, p+"_actor_p.h5")
    tf.keras.models.save_model(Agent.critic, p+"_critic.h5")
    print("Model Saved")
    return     

state = env.reset()
Agent = PPONetwork()
episode = 0
episode_reward = 0
score_history = []
test_score_hist  = []
test_std_hist = []
TOTAL_TR_STEPS = 100_000
start = time()
cofep = 0
EPISODES = 2000
step = 0
try:
    done = False
    for step in range(TOTAL_TR_STEPS):
        if step % 1000 == 0:
            #save_model()
            print("Step no: " + str(step))
        
        action_index, action_vector, action_prob = Agent.get_action(state)
        next_state, reward, done, info = env.step(action_index)
        # THIS IS FOR ONE STEP LEARNING
        Agent.learn(state, action_index, reward, next_state, done)
        Agent.store_transition(state, action_vector, reward,
                               action_prob, next_state, done)
        
        episode_reward += reward
        cofep+=1
        state = next_state

        if done or step == TOTAL_TR_STEPS-1: 
            episode += 1
            # THIS IS FOR EPISODIC LEARNING
            #Agent.train()
            score_history.append(episode_reward)
            avg_score = np.mean(score_history[-100:])
            print(f"At step: {step}")
            print('episode ', episode, 'score %.2f average score %2f' % \
                  (episode_reward, avg_score))
            episode_reward = 0
            t2 = time()
            print(f"time of ep: {t2-start} with {cofep} steps")
            start = time()
            cofep=0
            if (episode % 20 == 0 and done) or step == TOTAL_TR_STEPS-1:
                Mean, Var = test(Agent, env)
                test_score_hist.append(Mean)
                test_std_hist.append(Var)
            if done:
                next_state = env.reset()
                #save_model()
        state = next_state

except KeyboardInterrupt:
  #  save_model()
    raise
    
env.close()
#save_model()
#print(f"TOTAL TIME: {time()-t1}")
#save_model()