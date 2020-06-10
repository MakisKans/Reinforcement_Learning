#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 01:12:00 2019

@author: chryskan
"""
from dddqn_per import Agent, initial_observing
from params import *
from utilities import test_agent
if __name__=='__main__':
    agent = Agent()
    initial_observing(agent, env)

    episode_reward = 0
    score_history = []
    episode = 0
    Qs_history = []
    state = env.reset()

    try:
        for t in range(1, TOTAL_TR_STEPS+1):
            action = agent.predict_action(state)
            
            next_state, reward , done , info = env.step(action)
            episode_reward += reward
            experience = (state, action, reward, next_state, int(done))
            agent.memory.append(experience)
            state = next_state
            
            if t % TRAIN_EVERY == 0:
                loss = agent.train(step=t)
            if t % TARGET_UPDATE == 0:
                agent.update_target()            
            if t % TEST_EVERY == 0:
                _,_ = test_agent(agent)
                
            if done \
            or t==TOTAL_TR_STEPS-1\
            or (t % MAX_STEPS == 0 and t>0):
                episode+=1
                score_history.append(episode_reward)
                avg_score = np.mean(score_history[-20:])
                print("At step: ", t)
                print('episode ', episode, 'score %.2f average score %.2f' % \
                  (episode_reward, avg_score), 
                  'epsilon %.2f' % agent.epsilon)
                episode_reward = 0
                state = env.reset()
                temp = np.expand_dims(state.__array__(), axis=0)
                Qs = agent.q_net.predict(temp)
                Qs_history.append(np.max(Qs, axis=-1))
            
    except KeyboardInterrupt:
        agent.save()
        raise
    
    #agent.save()
    env.close()
