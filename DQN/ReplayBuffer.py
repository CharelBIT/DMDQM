import os
import sys
import cv2
import logging
import numpy as np
from collections import deque

import pickle
import decimal
import random


class ReplayBuffer:
    def __init__(self, config, model_dir):
        self.buffer_size=config.buffer_size
        self.model_dir = model_dir
        self.coor_action_size = config.coor_action_size
        self.radian_action_size = config.radian_action_size
        # self.train_action=config.train_action
        self.action_overturn = config.action_overturn
        self.squeues = 0
        self.image_per_squeues = 0
        self.cnn_format = config.cnn_format
        self.dataset_dir = config.dataset_dir
        self.memory_size = self.get_memorysize()
        self.buffer = deque()
        self.camera_num = config.camera_id
        self.num_experience=0
        self.screens_t = np.empty((self.camera_num+self.history_length+[config.screen_height, config.screen_width]), dtype=np.float16)
        self.screens_t_plus_1=np.empty((self.camera_num+self.history_length+[config.screen_height, config.screen_width]), dtype=np.float16)
        self.history_length = config.history_length
        self.dims = (config.screen_height, config.screen_width)
        self.batch_size = config.batch_size

    def add(self, screen,  coor_action, radian_action, coor_reward, radian_reward,screen_t_plus_1,terminal):
        experience=[screen,coor_action,radian_action,coor_reward,radian_reward,screen_t_plus_1,terminal]
        if self.num_experience<self.buffer_size:
            self.buffer.append(experience)
            self.num_experience+=1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)



    def sample(self):
        minibatch= random.sample(self.buffer,self.batch_size)
        if self.action_overturn:
            action_flags=np.randint(0,2,self.batch_size)
            for i in len(list(action_flags)):
                if action_flags[i]:
                    temp=minibatch[i][0]
                    minibatch[i][0]=minibatch[i][5]
                    minibatch[i][5]=temp
                    minibatch[i][1]=self.coor_action_size-1-minibatch[i][1]
                    minibatch[i][2]=self.radian_action_size-1-minibatch[i][2]
                    minibatch[i][3]=-minibatch[i][3]
                    minibatch[i][4] = -minibatch[i][4]
                    minibatch[i][6]=0
        return minibatch

    def save(self):
        if not os.path.exists('ReplayBuffer/'):
            os.makedirs('ReplayBuffer/')
        with open('ReplayBuffer_%s' % (self.num_experienced),'w') as f:
            pickle.dump(self.buffer,f)
            print("Save ReplayBuffer to disk...")

    def load(self):
        print("Load ReplayBuffer...")
        files=os.listdir('ReplayBuffer/')
        files=sorted(files,key=lambda x:int(x.split('_')[-1]))
        self.buffer=pickle.load(files[-1])
        print("Done!")





    def load_memory(self):
        squeues=os.listdir(self.dataset_dir+'/image')
        squeues=sorted(squeues,key=lambda x:int(x))
        for squeue in squeues:
            coor_reward_lines=open(self.dataset_dir+'/coor_reward/%s.txt' % (squeue),'rb').readlines()
            radian_reward_lines=open(self.dataset_dir+'/radian_reward/%s.txt' % (squeue),'rb').readlines()
            coor_action_lines=open(self.dataset_dir+'/coor_action/%s.txt' % (squeue),'rb').readlines()
            radian_action_lines=open(self.dataset_dir+'/radian_action/%s.txt' % (squeue),'rb').readlines()
            terminate_lines=open(self.dataset_dir+'/terminate/%s.txt' % (squeue),'rb').readlines()

            files=os.listdir(self.dataset_dir+'/image/%s' % (squeue))
            files=sorted(files,key=lambda x:int(x.split('.')[0]))
            for i in xrange(len(files)-self.history_length+1):
                for j in xrange(self.camera_num):
                    for k in xrange(self.history_length):
                        image_t=cv2.imread(self.dataset_dir+'/image/%s/camera%s/%s' % (squeue,j,files[i+k]),0)
                        image_t_plus_1 = cv2.imread(self.dataset_dir + '/image/%s/camera%s/%s' % (squeue, j, files[i + k+1]), 0)
                        image_t=cv2.resize(image_t,self.dims)
                        image_t_plus_1 = cv2.resize(image_t_plus_1, self.dims)
                        self.screens_t[j,k,...]=image_t
                        self.screens_t_plus_1[j,k,...]=image_t_plus_1
                coor_reward=float(coor_reward_lines[i+self.history_length-1])
                radian_reward=float(radian_reward_lines[i+self.history_length-1])

                coor_action=int(coor_action_lines[i+self.history_length-1])
                radian_action=int(radian_action_lines[i+self.history_length-1])
                terminate=bool(int(terminate_lines[i+self.history_length-1]))
                experience=[self.screens_t,coor_action,radian_action,coor_reward,radian_reward,self.screens_t_plus_1,terminate]
                self.buffer.append(experience)
                self.num_experience+=1

    def get_memorysize(self):
        total_state=0
        squeues=os.listdir(self.dataset_dir+'/image')
        squeues=sorted(squeues,key=lambda x:int(x))
        for squeue in squeues:
            image=os.listdir(self.dataset_dir+'/image'+squeue+'/camera0')
            total_state+=len(image)-3
        return total_state






