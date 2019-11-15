#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:10:34 2019

@author: chryskan
"""
from agent import *
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from utilities import *
from params import *
from test_PG import test
from time import time

cwd = os.getcwd() 
path = cwd+"/models/space_invaders/"
os.makedirs(path, exist_ok=True)
means , stds = [], []
agent = None

def save_model(agent):
    p = os.path.join(path, agent.name +"checkpoint.h5")
    agent.model.save(p)
    print("Model Saved")
    return

def main():  
    
  #  config = tf.ConfigProto()
   # config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
   # config.log_device_placement = True 
    # to log device placement (on which device the operation ran)
    # (nothing gets printed in Jupyter, only if you run itstandalone)
    #os.environ["CUDA_VISIBLE_DEVICES"]="0"
    
    #sess = tf.Session(config=config)
    #set_session(sess)
   
    # Avoid warning message errors

    # Allowing GPU memory growth
   
    Agent = PGNetwork()
    
    all_losses = []
    score_history = []
    
    mean_scrs, vars_scrs = [], []
    try:
        for step in range(5000):
            start = time()
            episode_reward = Agent.run_episode(env)
            score_history.append(episode_reward)
            avg_score = np.mean(score_history[-100:])
            print(f"At episode: {step}")
            print('tr score %.2f average score %2f' % (episode_reward, avg_score))
            UPDATE_FREQ = 4
            if step % UPDATE_FREQ == 0:
                loss = Agent.train()       
                if loss is not None:
                    all_losses.append(loss)
                Mean, Var = test(Agent,  env)
                mean_scrs.append(Mean)
                vars_scrs.append(Var)
            t2 = time()
            print(f"time of ep: {t2-start} ")

        #    if done or step == 99:
        #        all_rewards.append(episode_reward)
         #       state = env.reset()
                #stack = deque([np.zeros((84, 84), dtype= np.int) \
                 #              for i in range(STACK_SIZE)], maxlen= 4)
             #   state, stack = stack_frames(None, state, True)
    except KeyboardInterrupt:
      #  save_model(Agent)
      return mean_scrs, vars_scrs, Agent
      raise
        
    env.close()
    return mean_scrs, vars_scrs, Agent
   # save_model(Agent)
  #  with open("{}_train_res.pkl".format(Agent.name), "wb") as f:
   #     pkl.dump(all_rewards, f)
   #     pkl.dump(all_losses, f)

if __name__ == '__main__':
    means, stds, agent = main()