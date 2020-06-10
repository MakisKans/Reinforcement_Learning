#!/usr/bin/env python4
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 17:24:19 2019

@author: chryskan
"""
import tensorflow as tf
import numpy as np
from collections import deque
from params import *
import gym
from atari_wrappers import make_atari, wrap_deepmind

def lin_epsilon(step, start=E_START, stop=E_STOP, steps = DECAY_STEPS):
    return (stop - start)*step/steps + start

class ImageTransformer():
    def __init__(self):
        with tf.variable_scope("image_transformer"):
          self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
          self.output = tf.image.rgb_to_grayscale(self.input_state)
          self.output = self.output/255
          self.output = tf.image.resize_images(
                  self.output, [height, width],
                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
     #     self.output = tf.image.crop_to_bounding_box(
      #            self.output, 13, 0, 84, 84)
          self.output = tf.squeeze(self.output)

    def transform(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.output, { self.input_state: state })

def update_stack(stack, obs, is_new=False):
    if is_new:
        stack = np.zeros_like(np.stack([obs]*4, axis=-1))
    np.append(stack[:,:,1:], np.expand_dims(obs, -1), axis=-1)
    return stack

def stack_frames(stack, state, is_new):
    if is_new:
        stack = deque([np.zeros((84, 84), dtype= np.int) for i in
                       range(STACK_SIZE)], maxlen=STACK_SIZE)
        for i in range(STACK_SIZE):
            stack.append(state)
    else:
        stack.append(state)

    stacked_state = np.stack(stack, axis=2)
    return stacked_state, stack

def random_noops(env):
    state = env.reset()
    noops = np.random.randint(NOOP+1)
    for i in range(noops):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        if done:
            state = env.reset()
    return state

RENDER = False

def test_agent(agent):
    test_history = []
    n_games = 0

    env = wrap_deepmind(make_atari(GAME), episode_life=True, clip_rewards=True, 
                  frame_stack=True, scale=True)
    state = env.reset().__array__()
    score = 0

    for step in range(TEST_STEPS):
        Qs = agent.q_net.predict(np.expand_dims(state, axis=0))
        choice = np.argmax(Qs)
        state_, reward, done, info = env.step(choice)
        state = state_.__array__()
        score += reward
        if done:
            n_games+=1
            test_history.append(score)
            state = env.reset().__array__()
            score = 0
    print("**********\nTEST\naverage score: ", np.mean(test_history),
                  "\nstd deviation: ", np.std(test_history),"\n**********")
    env.close()
    return  np.mean(test_history), np.std(test_history)
