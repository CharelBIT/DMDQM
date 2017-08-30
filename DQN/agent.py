from __future__ import print_function
import os
import sys
import cv2
import time
import random
import math
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from DQN.history import History
from tools.Preprocess import decimalToAny

from DQN.base import BaseModel
from DQN.replay_memory import ReplayMemory
from DQN import ReplayBuffer
from DQN.ops import linear, conv2d, clipped_error
from DQN.utils import get_time, save_pkl, load_pkl
from pexpect import expect


def euclideandistance(line1, line2):
    assert len(line1) == len(line2)
    distance = 0.0
    for i in range(len(line1)):
        distance = math.pow((line1[i] - line2[i]), 2) + distance
    return math.sqrt(distance)

class Agent(BaseModel):
  def __init__(self, config,env, sess):
    super(Agent, self).__init__(config)
    self.sess = sess
    self.camera_id=config.camera_id
    self.explore_iter=config.explore_iter
    self.explore_test_iter=config.explore_test_iter
    self.explore_train_iter=config.explore_train_iter
    self.weight_dir = 'weight'
    self.radian_coor_coef=config.radian_coor_coef
    self.coor_action_size=config.coor_action_size
    self.radian_action_size=config.radian_action_size
#    self.action_overturn=config.action_overturn
    self.mode_path=config.model_path

    self.coor_position=[]
    self.radian_position=[]

    self.pre_coor_position=[]
    self.pre_radian_position=[]

    self.post_coor_position=[]
    self.post_radian_position=[]

    self.history=History(config)
    self.is_train=config.is_train
    self.env=env
    self.memory = ReplayBuffer(self.config, self.model_dir)
    self.channel=self.memory.history_length*self.memory.camera_num
    if self.is_train:
        self.memory.load_memory()

    with tf.variable_scope('step'):
      self.step_op = tf.Variable(0, trainable=False, name='step')
      self.step_input = tf.placeholder('int32', None, name='step_input')
      self.step_assign_op = self.step_op.assign(self.step_input)

    self.build_dqn()

  def explore(self):
      for exp_iter in range(self.explore_iter):
        print('\rExploring %s iters' % (exp_iter))
        self.train(max_step=self.explore_train_iter)
        self.test(max_step=self.explore_test_iter)
        self.save_weight_to_pkl(exp_iter,phase='explore')
        lines=raw_input('Please change the postion of object,and input new position(coor and radian):')
        lines=lines.split(' ')
        assert lines==6
        self.coor_position=lines[:3]
        self.radian_position=lines[3:6]
        for i in xrange(3):
            self.coor_position[i]=float(self.coor_position[i])
            self.radian_position[i] = float(self.radian_position[i])


  def test(self,max_step=None):
      print('\rTesting')
      for step in xrange(max_step):
          cap=[]
          screens_t = np.empty((self.camera_num +self.memory.history_length+self.memory.dims), dtype=np.float16)
          screens_t_plus_1 = np.empty((self.camera_num +self.memory.history_length+self.memory.dims),dtype=np.float16)
          self.env.init_position()
          self.pre_coor_position,self.pre_radian_position=self.env.getPose()
          for i in range(self.camera_id):
              cap[i]=cv2.VideoCapture(i)
          cv2.waitKey(10)
          for i in range(self.memory.history_length):
              for j in range(self.camera_id):
                  ret,frame=cap[j].read()
                  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                  frame=cv2.resize(frame,self.memory.dims)
                  screens_t[j,i,...]=frame
              coor_action=np.random.randint(0,self.memory.coor_action_size)
              radian_action=np.random.randint(0,self.memory.radian_action_size)
              self.env.move(coor_action,radian_action)
              for k in range(self.camera_id):
                  ret,frame=cap[j].read()
                  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                  frame=cv2.resize(frame,self.memory.dims)
                  screens_t_plus_1[k,i,...]=frame
          self.post_coor_position,self.post_radian_position=self.env.getPose()
          coor_reward,radian_reward=self.gen_reward()
          terminal=0
          self.memory.add(screens_t,coor_action,radian_action,float(coor_reward),float(radian_reward),
                          screens_t_plus_1,terminal)
          screens_t=screens_t_plus_1
          self.pre_radian_position=self.post_radian_position
          self.pre_coor_position=self.post_coor_position
          while True:
              coor_action, radian_action = self.sess.run([self.coor_q_action, self.radian_q_action],
                                                         {self.s_t: [screens_t.reshape(self.channel+self.memory.dims)]})
              for i in xrange(self.camera_num):
                screens_t_plus_1[i,-1,...]=cap[i].read()
              self.post_coor_position,self.post_radian_position=self.env.getPose()
              coor_reward,radian_reward=self.gen_reward()
              if not self.env.is_safe():
                  terminal=1
                  self.memory.add(screens_t, coor_action, radian_action, float(coor_reward), float(radian_reward),
                            screens_t_plus_1,terminal)
                  break
              else:
                  terminal=0
                  self.memory.add(screens_t, coor_action, radian_action, float(coor_reward), float(radian_reward),
                            screens_t_plus_1,terminal)
              screens_t=screens_t_plus_1
              self.pre_radian_position=self.post_radian_position
              self.pre_coor_position=self.post_coor_position

  def gen_reward(self):
      pre_coor_distance = euclideandistance(self.pre_coor_position, self.coor_position)
      pre_radian_distance = euclideandistance(self.pre_radian_position, self.radian_position)

      post_coor_distance = euclideandistance(self.post_coor_position, self.coor_position)
      post_radian_distance = euclideandistance(self.post_radian_position, self.radian_position)

      return np.sign(pre_coor_distance-post_coor_distance),np.sign(pre_radian_distance-post_radian_distance)

  def train(self,max_step=None):
    if max_step==None:
        max_step=self.max_step
    start_step = self.step_op.eval()
    self.update_count=  0
    self.sample_coor_actions=[]
    self.sample_radian_actions=[]
    self.sample_coor_rewards=[]
    self.sample_radian_rewards=[]
    self.total_reward, self.total_loss, self.total_q = 0., 0., 0.
    self.total_coor_loss,self.total_radian_loss=0.,0.
    self.total_coor_reward,self.total_radian_reward=0.,0.
    self.total_coor_q_t,self.total_radian_q_t=0.,0.
    

    for step in tqdm(range(start_step, self.max_step), ncols=70, initial=start_step):
      if step == self.learn_start:
        self.total_loss, self.total_q = 0., 0.
        self.total_coor_loss,self.total_radian_loss=0. ,0.
        self.total_coor_q,self.total_radian_q=0. ,0.
        
      
      if step >= self.learn_start:
          
        self.q_learning_mini_batch()
        
        if step % self.target_q_update_step == self.target_q_update_step - 1:
          self.update_target_q_network()
          print("\rUpdate Tareget Net at %d\r" % (self.step))
          
        if self.step % self.snap_step == self.snap_step - 1:
          avg_reward =np.mean(np.array(self.total_reward/self.snap_step)) 
          avg_coor_reward=np.mean(np.array(self.total_coor_reward/self.snap_step))
          avg_radian_reward=np.mean(np.array(self.total_radian_reward/self.snap_step))  
          
          avg_loss = self.total_loss / self.snap_step
          
          avg_q = (self.total_q / self.snap_step).mean()
          
          
          avg_coor_loss =self.total_coor_loss/self.snap_step
          
          avg_radian_loss=self.total_radian_loss/self.snap_step
          
          avg_radian_q=(self.total_radian_q_t/self.snap_step).mean()
          
          
          avg_coor_q=(self.total_coor_q_t/self.snap_step).mean()
          
          print('\ravg_loss=%.4f,avg_radian_loss=%.4f,avg_coor_loss=%.4f' % (avg_loss,avg_radian_loss,avg_coor_loss))
          print('\ravg_q=%.4f,avg_radian_q=%.4f,avg_coor_q=%.4f' % (avg_q,avg_radian_q,avg_coor_q))
          print('\ravg_reward=%.4f,avg_radian_reward=%.4f,avg_coor_reward=%.4f' % (avg_reward,avg_radian_reward,avg_coor_reward))

          if self.step > 180:
            self.inject_summary({
                 'average.reward': avg_reward,
                 'average.coor_reward':avg_coor_reward,
                 'average.radian_reward':avg_radian_reward,
                 
                'average.loss': avg_loss,
                'average.coor_loss':avg_coor_loss,
                'average.radian_loss':avg_radian_loss,
                'average.q': avg_q,
                'average.coor q': avg_coor_q,
                'average.radian q': avg_radian_q,
                'training.learning_rate': self.learning_rate_op.eval({self.learning_rate_step: self.step}),
              }, self.step)
          self.total_reward=0.
          self.total_loss, self.total_q = 0., 0.
          self.total_coor_loss,self.total_radian_loss=0. ,0.
          self.total_coor_q_t,self.total_radian_q_t=0. ,0.
          self.total_coor_reward,self.total_radian_reward=0.,0.
        if self.step % self.save_iter==self.save_iter-1:
          self.step_assign_op.eval({self.step_input: self.step})
          self.save_weight_to_pkl(self.step)

  def predict(self, s_t, test_ep=None):
    coor_action,radian_action=self.sess.run([self.coor_q_action,self.radian_q_action],{self.s_t:[s_t]})


    return coor_action,radian_action

  def q_learning_mini_batch(self):

    # s_t,coor_action,radian_action,coor_reward,radian_reward,s_t_plus_1,terminal= self.memory.sample()
    minibatch=self.memory.sample()
    s_t=np.asarray([data[0] for data in minibatch])
    coor_action=np.asarray([data[1] for data in minibatch])
    radian_action = np.asarray([data[2] for data in minibatch])
    coor_reward=np.asarray([data[3] for data in minibatch])
    radian_reward = np.asarray([data[4] for data in minibatch])
    s_t_plus_1=np.asarray([data[5] for data in minibatch])
    terminal = np.asarray([data[6] for data in minibatch])
    s_t=s_t.reshape((self.channel)+self.memory.dims)
    s_t_plus_1 = s_t_plus_1.reshape((self.channel) + self.memory.dims)
#     for i in range(self.memory.batch_size):
#         self.sample_coor_actions[coor_action[i]]+=1
#         self.sample_radian_actions[radian_action[i]]+=1
    if self.double_q:
      # Double Q-learning
      pred_coor_action = self.coor_q_action.eval({self.s_t: s_t_plus_1})
      pred_radian_action = self.radian_q_action.eval({self.s_t: s_t_plus_1})

      q_t_plus_1_with_pred_coor_action = self.target_coor_q_with_idx.eval({
        self.target_s_t: s_t_plus_1,
        self.target_coor_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_coor_action)]
      })
      
      q_t_plus_1_with_pred_radian_action = self.target_radian_q_with_idx.eval({
        self.target_s_t: s_t_plus_1,
        self.target_radian_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_radian_action)]
      })
      
      target_coor_q_t = (1. - terminal) * self.discount * q_t_plus_1_with_pred_coor_action + coor_reward
      target_radian_q_t = (1. - terminal) * self.discount * q_t_plus_1_with_pred_radian_action + radian_reward
    else:
      coor_q_t_plus_1 = self.target_coor_q.eval({self.target_s_t: s_t_plus_1})
      radian_q_t_plus_1 = self.target_coor_q.eval({self.target__s_t: s_t_plus_1})
      

      terminal = np.array(terminal) + 0.
      max_coor_q_t_plus_1 = np.max(coor_q_t_plus_1, axis=1)
      max_radian_q_t_plus_1 = np.max(radian_q_t_plus_1, axis=1)
      
      target_coor_q_t = (1. - terminal) * self.discount * max_coor_q_t_plus_1 + coor_reward
      target_radian_q_t = (1. - terminal) * self.discount * max_radian_q_t_plus_1 + radian_reward
      

    _, coor_q_t, radian_q_t,coor_loss,radian_loss,loss, summary_str, summary_str_action = \
    self.sess.run([self.optim, self.coor_q,self.radian_q,self.coor_loss,self.radian_loss, self.loss, self.q_summary,self.summary_action],{ \
      self.target_coor_q_t: target_coor_q_t, \
      self.target_radian_q_t: target_radian_q_t, \
      self.coor_action: coor_action, \
      self.radian_action: radian_action, \
      self.s_t: s_t, \
      self.learning_rate_step: self.step, \
      self.histogram_placeholders['coor_actions']: coor_action, \
      self.histogram_placeholders['radian_actions']: radian_action})
    
    self.writer.add_summary(summary_str, self.step)
    self.writer.add_summary(summary_str_action, self.step)
    self.total_coor_loss +=coor_loss
    self.total_radian_loss +=radian_loss
    self.total_loss += loss
    self.total_coor_q_t+=coor_q_t
    self.total_radian_q_t+=radian_q_t
    
    self.total_q +=(coor_q_t+radian_q_t)
    self.total_coor_reward+=coor_reward
    self.total_radian_reward+=radian_reward
    self.total_reward+=(coor_reward+radian_reward)


  def build_dqn(self):
    self.w = {}
    self.t_w = {}

    #initializer = tf.contrib.layers.xavier_initializer()
    initializer = tf.truncated_normal_initializer(0, 0.02)
    activation_fn = tf.nn.relu

    # training network
    with tf.variable_scope('prediction'):
      if self.cnn_format == 'NHWC':
        self.s_t = tf.placeholder('float32',
            [None, self.screen_height, self.screen_width, self.channel], name='s_t')
        self.y_test=tf.placeholder(tf.int64,[None,1],name='y_test')
      else:
        self.s_t = tf.placeholder('float32',
            [None, self.channel, self.screen_height, self.screen_width], name='s_t')
        self.y_test=tf.placeholder(tf.int64,[None,1],name='y_test')

      self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.s_t,
          32, [8, 8], [4, 4], initializer, activation_fn, self.cnn_format, name='l1')
      self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1,
          64, [4, 4], [2, 2], initializer, activation_fn, self.cnn_format, name='l2')
      self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2,
          64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='l3')

      shape = self.l3.get_shape().as_list()
      self.l3_flat = tf.reshape(self.l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

      if self.dueling:
        self.coor_value_hid, self.w['coor_l4_val_w'], self.w['coor_l4_val_b'] = \
            linear(self.l3_flat, 512, activation_fn=activation_fn, name='coor_value_hid')

        self.coor_adv_hid, self.w['coor_l4_adv_w'], self.w['coor_l4_adv_b'] = \
            linear(self.l3_flat, 512, activation_fn=activation_fn, name='coor_adv_hid')

        self.coor_value, self.w['coor_val_w_out'], self.w['coor_val_w_b'] = \
          linear(self.coor_value_hid, 1, name='coor_value_out')

        self.coor_advantage, self.w['coor_adv_w_out'], self.w['coor_adv_w_b'] = \
          linear(self.coor_adv_hid, self.coor_action_size, name='coor_adv_out')

        # Average Dueling
        self.coor_q = self.coor_value + (self.coor_advantage - 
          tf.reduce_mean(self.coor_advantage, reduction_indices=1, keep_dims=True))

        self.radian_value_hid, self.w['radian_l4_val_w'], self.w['radian_l4_val_b'] = \
            linear(self.l3_flat, 512, activation_fn=activation_fn, name='radian_value_hid')

        self.radian_adv_hid, self.w['radian_l4_adv_w'], self.w['radian_l4_adv_b'] = \
            linear(self.l3_flat, 512, activation_fn=activation_fn, name='radian_adv_hid')

        self.radian_value, self.w['radian_val_w_out'], self.w['radian_val_w_b'] = \
          linear(self.radian_value_hid, 1, name='radian_value_out')

        self.radian_advantage, self.w['radian_adv_w_out'], self.w['radian_adv_w_b'] = \
          linear(self.radian_adv_hid, self.radian_action_size, name='radian_adv_out')

        # Average Dueling
        self.radian_q = self.radian_value + (self.radian_advantage - 
          tf.reduce_mean(self.radian_advantage, reduction_indices=1, keep_dims=True))
      else:
        self.coor_l4, self.w['coor_l4_w'], self.w['coor_l4_b'] = linear(self.l3_flat, 512, activation_fn=activation_fn, name='coor_l4')
        self.coor_q, self.w['coor_q_w'], self.w['coor_q_b'] = linear(self.coor_l4, self.coor_action_size, name='coor_q')
        
        self.radian_l4, self.w['radian_l4_w'], self.w['radian_l4_b'] = linear(self.l3_flat, 512, activation_fn=activation_fn, name='radian_l4')
        self.radian_q, self.w['radian_q_w'], self.w['radian_q_b'] = linear(self.radian_l4, self.coor_action_size, name='radian_q')

      self.coor_q_action = tf.argmax(self.coor_q, dimension=1)
      self.radian_q_action = tf.argmax(self.radian_q, dimension=1)

      q_summary = []
      avg_coor_q = tf.reduce_mean(self.coor_q, 0)
      avg_radian_q=tf.reduce_mean(self.radian_q, 0)
      for idx in xrange(self.coor_action_size):
        q_summary.append(tf.summary.histogram('coor_q/%s' % idx, avg_coor_q[idx]))
        
      for idx in xrange(self.radian_action_size):
        q_summary.append(tf.summary.histogram('radian_q/%s' % idx, avg_radian_q[idx]))
      
      self.q_summary = tf.summary.merge(q_summary, 'q_summary')

    # target network
    with tf.variable_scope('target'):
      if self.cnn_format == 'NHWC':
        self.target_s_t = tf.placeholder('float32', 
            [None, self.screen_height, self.screen_width, self.channel], name='target_s_t')
      else:
        self.target_s_t = tf.placeholder('float32', 
            [None, self.channel, self.screen_height, self.screen_width], name='target_s_t')

      self.target_l1, self.t_w['l1_w'], self.t_w['l1_b'] = conv2d(self.target_s_t, 
          32, [8, 8], [4, 4], initializer, activation_fn, self.cnn_format, name='target_l1')
      self.target_l2, self.t_w['l2_w'], self.t_w['l2_b'] = conv2d(self.target_l1,
          64, [4, 4], [2, 2], initializer, activation_fn, self.cnn_format, name='target_l2')
      self.target_l3, self.t_w['l3_w'], self.t_w['l3_b'] = conv2d(self.target_l2,
          64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='target_l3')

      shape = self.target_l3.get_shape().as_list()
      self.target_l3_flat = tf.reshape(self.target_l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

      if self.dueling:
        self.t_coor_value_hid, self.t_w['coor_l4_val_w'], self.t_w['coor_l4_val_b'] = \
            linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_coor_value_hid')

        self.t_coor_adv_hid, self.t_w['coor_l4_adv_w'], self.t_w['coor_l4_adv_b'] = \
            linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_coor_adv_hid')

        self.t_coor_value, self.t_w['coor_val_w_out'], self.t_w['coor_val_w_b'] = \
          linear(self.t_coor_value_hid, 1, name='target_coor_value_out')

        self.t_coor_advantage, self.t_w['coor_adv_w_out'], self.t_w['coor_adv_w_b'] = \
          linear(self.t_coor_adv_hid, self.coor_action_size, name='target_coor_adv_out')

        # Average Dueling
        self.target_coor_q = self.t_coor_value + (self.t_coor_advantage - 
          tf.reduce_mean(self.t_coor_advantage, reduction_indices=1, keep_dims=True))
        
        self.t_radian_value_hid, self.t_w['radian_l4_val_w'], self.t_w['radian_l4_val_b'] = \
            linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_radian_value_hid')

        self.t_radian_adv_hid, self.t_w['radian_l4_adv_w'], self.t_w['radian_l4_adv_b'] = \
            linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='radian_coor_adv_hid')

        self.t_radian_value, self.t_w['radian_val_w_out'], self.t_w['radian_val_w_b'] = \
          linear(self.t_radian_value_hid, 1, name='target_radian_value_out')

        self.t_radian_advantage, self.t_w['radian_adv_w_out'], self.t_w['radian_adv_w_b'] = \
          linear(self.t_radian_adv_hid, self.radian_action_size, name='target_radian_adv_out')

        # Average Dueling
        self.target_radian_q = self.t_radian_value + (self.t_radian_advantage - 
          tf.reduce_mean(self.t_radian_advantage, reduction_indices=1, keep_dims=True))
        
        
      else:
        self.target_coor_l4, self.t_w['coor_l4_w'], self.t_w['coor_l4_b'] = \
            linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='coor_target_l4')
        self.target_coor_q, self.t_w['coor_q_w'], self.t_w['coor_q_b'] = \
            linear(self.target_coor_l4, self.action_size, name='coor_target_q')

        self.target_radian_l4, self.t_w['radian_l4_w'], self.t_w['radian_l4_b'] = \
            linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='radian_target_l4')
        self.target_radian_q, self.t_w['radian_q_w'], self.t_w['radian_q_b'] = \
            linear(self.target_radian_l4, self.radian_action_size, name='radian_target_q')
            
      self.target_coor_q_idx = tf.placeholder('int32', [None, None], 'coor_outputs_idx')
      self.target_coor_q_with_idx = tf.gather_nd(self.target_coor_q, self.target_coor_q_idx)
      
      self.target_radian_q_idx = tf.placeholder('int32', [None, None], 'radian_outputs_idx')
      self.target_radian_q_with_idx = tf.gather_nd(self.target_radian_q, self.target_radian_q_idx)
      
    with tf.variable_scope('pred_to_target'):
      self.t_w_input = {}
      self.t_w_assign_op = {}

      for name in self.w.keys():
        self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
        self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

    # optimizer
    with tf.variable_scope('optimizer'):
      self.target_coor_q_t = tf.placeholder('float32', [None], name='target_coor_q_t')
      self.target_radian_q_t = tf.placeholder('float32', [None], name='target_radian_q_t')
      
      self.coor_action = tf.placeholder('int64', [None], name='coor_action')
      self.radian_action = tf.placeholder('int64', [None], name='radian_action')

      coor_action_one_hot = tf.one_hot(self.coor_action, self.coor_action_size, 1.0, 0.0, name='coor_action_one_hot')
      coor_q_acted = tf.reduce_sum(self.coor_q * coor_action_one_hot, reduction_indices=1, name='coor_q_acted')
      
      radian_action_one_hot = tf.one_hot(self.radian_action, self.radian_action_size, 1.0, 0.0, name='radian_action_one_hot')
      radian_q_acted = tf.reduce_sum(self.radian_q * radian_action_one_hot, reduction_indices=1, name='radian_q_acted')

      self.coor_delta = self.target_coor_q_t - coor_q_acted
      
      self.radian_delta = self.target_radian_q_t - radian_q_acted
#      self.delta=self.coor_delta*self.radian_coor_coef+self.radian_delta*(1-self.radian_coor_coef)
      

      self.global_step = tf.Variable(0, trainable=False)

      self.coor_loss = tf.reduce_mean(clipped_error(self.coor_delta), name='coor_loss')
      
      self.radian_loss=tf.reduce_mean(clipped_error(self.radian_delta), name='radian_loss')
      self.loss=self.coor_loss*self.radian_coor_coef+self.radian_loss*(1-self.radian_coor_coef)
      self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
      self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
          tf.train.exponential_decay(
              self.learning_rate,
              self.learning_rate_step,
              self.learning_rate_decay_step,
              self.learning_rate_decay,
              staircase=True))
      self.optim = tf.train.RMSPropOptimizer(
          self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)
          
#     with tf.variable_scope('test'):
#         correct_prediction=tf.equal(self.q_action,self.y_test)
#         self.test_accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    with tf.variable_scope('summary'):
#       scalar_summary_tags = ['average.reward', 'average.loss', 'average.q', \
#           'episode.max reward', 'episode.min reward', 'episode.avg reward', 'episode.num of game', 'training.learning_rate']
      scalar_summmary_tags=['average.reward','average.coor_reward','average.radian_reward','average.loss','average.coor_loss',
                            'average.radian_loss','average.q','average.coor q','average.radian q','training.learning_rate']
      self.summary_placeholders = {}
      self.summary_ops = {}

      for tag in scalar_summmary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag]  = tf.summary.scalar("%s-%s/%s" % (self.env_name, self.env_type, tag), self.summary_placeholders[tag])

      histogram_summary_tags = ['coor_actions','radian_actions']

      summary_action=[]
      self.histogram_placeholders={}
      self.summary_action={}
      for tag in histogram_summary_tags:
        self.histogram_placeholders[tag] = tf.placeholder('int8', None, name=tag.replace(' ', '_'))
        self.summary_action[tag]  = tf.summary.histogram(tag, self.histogram_placeholders[tag])
        summary_action.append(self.summary_action[tag])
      self.summary_action=tf.summary.merge(summary_action,'pred_action')

      self.writer = tf.summary.FileWriter('./logs', self.sess.graph)

    init=tf.global_variables_initializer()
    self.sess.run(init)
    if self.is_train:
        checks=os.listdir('checkpoints')
        checks=sorted(checks,key=lambda x:int(x))
   #     self.load_model(int(checks[-1]))
    else:
        weights=os.listdir('weight')
        weights=sorted(weights,key=lambda x:int(x))
        self.load_weight_from_pkl(int(weights[-1]))

 #   self._saver = tf.train.Saver(self.w.values() + [self.step_op], max_to_keep=30)

#    self.load_model(230000)
    self.update_target_q_network()

  def update_target_q_network(self):
    for name in self.w.keys():
      self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})

  def save_weight_to_pkl(self,step,phase='explore'):
    if not os.path.exists(self.weight_dir):
        os.makedirs(self.weight_dir)
    if not os.path.exists(os.path.join(self.weight_dir,phase+'_%s' % (step))):
      os.makedirs(os.path.join(self.weight_dir,phase+'_%s' % (step)))

    for name in self.w.keys():
      save_pkl(self.w[name].eval(), os.path.join(self.weight_dir+'/'+str(step), "%s.pkl" % name))

  def load_weight_from_pkl(self, step,cpu_mode=False):
    with tf.variable_scope('load_pred_from_pkl'):
      self.w_input = {}
      self.w_assign_op = {}

      for name in self.w.keys():
        self.w_input[name] = tf.placeholder('float32', self.w[name].get_shape().as_list(), name=name)
        self.w_assign_op[name] = self.w[name].assign(self.w_input[name])

    for name in self.w.keys():
      self.w_assign_op[name].eval({self.w_input[name]: load_pkl(os.path.join(self.weight_dir+'/'+str(step), "%s.pkl" % name))})

    self.update_target_q_network()

  def inject_summary(self, tag_dict, step):
    summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
      self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
    })
    for summary_str in summary_str_lists:
      self.writer.add_summary(summary_str, self.step)
    
  def play(self, n_step=10000, n_episode=100, test_ep=None, render=False):
    try:
        self.env.s.connect((self.HOST,self.PORT))
        time.sleep(1.00)
    except RuntimeError:
        print("Connecting Error")     
    cap=cv2.VideoCapture(1)
    cv2.namedWindow('capture')    
    ret,frame=cap.read()
    frame=cv2.resize(frame,self.env.dims)
    frame_gray=np.zeros(frame.shape,dtype=np.float16)
    frame_gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
    self.history.initial_history(frame_gray)
    while True:
        ret,frame=cap.read()
        start=time.clock()
        coor_action,radian_action=self.predict(self.history.history)
        sys.stdout.write('\rPrediction> > coor_action=%d,radian_action=%d' % (coor_action,radian_action))
        coor_result=decimalToAny(coor_action,3)
        radian_result=decimalToAny(radian_action, 3)
        self.env.getPose()
        self.env.move(coor_result,radian_result)
        end=(time.clock()-start)*1000
        
        
        cv2.imshow('capture',frame)
        
        frame=cv2.resize(frame,self.env.dims)
#        frame=np.array(frame,dtype=np.float16)
        frame_gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
        
        self.history.add(frame_gray)
#        self.history.initial_history(frame_gray)
        if end>=30:
            continue
        else:
            cv2.waitKey(30-int(end))
        
        
