import os
import sys
import cv2
import logging
import numpy as np

from DQN.utils import save_npy, load_npy
import decimal

class ReplayMemory:
  def __init__(self, config, model_dir):
    self.model_dir = model_dir
    self.coor_action_size=config.coor_action_size
    self.radian_action_size=config.radian_action_size
    #self.train_action=config.train_action
    self.action_overturn=config.action_overturn
    self.squeues=0
    self.image_per_squeues=0
    self.cnn_format = config.cnn_format
    self.dataset_dir=config.dataset_dir
    self.memory_size =self.get_memorysize()
    self.camera_num=config.camera_id
    
    self.coor_actions = np.empty((self.memory_size[0],self.memory_size[1]), dtype = np.uint8)
    self.radian_actions=np.empty((self.memory_size[0],self.memory_size[1]),dtype=np.uint8)
    
    self.coor_rewards = np.empty((self.memory_size[0],self.memory_size[1]), dtype = np.float32)
    self.radian_rewards = np.empty((self.memory_size[0],self.memory_size[1]), dtype = np.float32)
    
    self.screens = np.empty((self.memory_size+[config.screen_height, config.screen_width]), dtype = np.float16)
    self.terminals = np.empty((self.memory_size[0],self.memory_size[1]), dtype = np.bool)
    self.history_length = config.history_length
    self.dims = (config.screen_height, config.screen_width)
    self.batch_size = config.batch_size

    self.prestates = np.empty((self.batch_size,self.history_length, self.camera_num) + self.dims, dtype = np.float16)
    self.poststates = np.empty((self.batch_size,self.history_length, self.camera_num) + self.dims, dtype = np.float16)

  def add(self, screen, coor_reward,radian_reward, coor_action,radian_action,terminal):
    assert screen.shape == self.dims
    self.coor_actions[self.current] = coor_action
    self.radian_action[self.current]=radian_action
    
    self.coor_rewards[self.current] = coor_reward
    self.radian_rewards[self.current] = radian_reward
    
    self.screens[self.current, ...] = screen
    self.terminals[self.current] = terminal
    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.memory_size

  def sample(self):
    # range_squeues=[i for i in range(self.squeues)]
    # index_squeues=np.random.choice(range_squeues,self.batch_size)
    coor_actions,radian_actions=[],[]
    coor_rewards,radian_rewards=[],[]
    terminals=[]
    index_squeues=np.random.randint(0,self.squeues,self.batch_size)
    if self.action_overturn:
        random_action_overturn=np.random.randint(0,2,32)
    else:
        random_action_overturn=np.zeros(32,dtype=np.bool)
    for i in range(self.batch_size):
        if random_action_overturn[i]:
            index_pre_image=np.random.randint(self.history_length,self.image_per_squeues)
            index_post_image=index_pre_image-1
            self.prestates[i,...]=self.screens[index_squeues[i],index_post_image-self.history_length-1:index_post_image+1,...]
            self.poststates[i,...]=self.screens[index_squeues[i],index_pre_image-self.history_length-1:index_pre_image+1,...]

            coor_actions.append(27-self.coor_actions[index_squeues[i],index_pre_image])
            radian_actions.append(27-self.radian_actions[index_squeues[i],index_pre_image])

            coor_rewards.append(-1.0*self.coor_rewards[index_squeues[i],index_pre_image])
            radian_rewards.append(-1.0*self.radian_rewards[index_squeues[i],index_pre_image])
            terminals.append(0)
        else:
            index_pre_image=np.random.randint(self.history_length,self.image_per_squeues)
            if index_pre_image==self.image_per_squeues-1:
                self.prestates[i, ...] = self.screens[index_squeues[i],
                                         index_pre_image - self.history_length - 1:index_pre_image + 1, ...]
                self.poststates[i, ...] = self.screens[index_squeues[i],
                                          index_pre_image - self.history_length - 1:index_pre_image + 1, ...]
            else:
                index_post_image=index_pre_image+1
                self.prestates[i, ...] = self.screens[index_squeues[i],
                                         index_pre_image - self.history_length - 1:index_pre_image + 1, ...]
                self.poststates[i, ...] = self.screens[index_squeues[i],
                                          index_post_image - self.history_length - 1:index_post_image + 1, ...]

            coor_actions.append(self.coor_actions[index_squeues[i], index_pre_image])
            radian_actions.append(self.radian_actions[index_squeues[i], index_pre_image])

            coor_rewards.append(self.coor_rewards[index_squeues[i], index_pre_image])
            radian_rewards.append(self.radian_rewards[index_squeues[i], index_pre_image])

            terminals.append(self.terminals[index_squeues[i],index_pre_image])

    coor_actions=np.array(coor_actions).reshape(self.batch_size,1)
    radian_actions = np.array(radian_actions).reshape(self.batch_size, 1)

    radian_rewards = np.array(radian_rewards).reshape(self.batch_size, 1)
    coor_rewards = np.array(coor_rewards).reshape(self.batch_size, 1)

    terminals= np.array(terminals).reshape(self.batch_size, 1)

    if self.cnn_format == 'NHWC':
 
        return np.transpose(self.prestates, (0, 2, 3, 1)), coor_actions, radian_actions,\
            coor_rewards, radian_rewards,np.transpose(self.poststates, (0, 2, 3, 1)), terminals
    else:
        return self.prestates, coor_actions, radian_actions,coor_rewards, radian_rewards, self.poststates, terminals


  def save(self):
    for idx, (name, array) in enumerate(
        zip(['coor_actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates'],
            [self.coor_actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates])):
      save_npy(array, os.path.join(self.model_dir, name))

  def load(self):
    for idx, (name, array) in enumerate(
        zip(['coor_actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates'],
            [self.coor_actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates])):
      array = load_npy(os.path.join(self.model_dir, name))
      
  def load_image(self):
      sys.stdout.write('\rLoad memory>>Load Images from Dataset~~~~~~~~~~~~~~~~~~~')
      image_dir=self.dataset_dir+'/image'
      camera_dir=os.listdir(image_dir)
      camera_dir=sorted(camera_dir,key=lambda x:int(x))
      squeues_paths=os.listdir(image_dir+'/'+camera_dir[0])
      squeues_paths=sorted(squeues_paths,key=lambda x:int(x))
      for i in range(len(squeues_paths)):
          files=os.listdir(os.path.join(image_dir,squeues_paths[i]))
          files=sorted(files,key=lambda x:int(x.split('.')[0]))
          for j in range(len(files)):
              for k in range(self.camera_num):
                image=cv2.imread(image_dir+'/'+squeues_paths[i]+'/'+files[j],0)
                image=cv2.resize(image,(self.screens.shape[3],self.screens.shape[2]))
                self.screens[i,j,k,:,:]=np.array(image,dtype=np.float32)
              sys.stdout.write('\rLoad memory>>Load Images >> %s/%s squeues' % 
                                (files[j],squeues_paths[i]))
              
  def load_action_terminate(self,action_type):
      sys.stdout.write('\rLoad memory>>Load Action from Dataset~~~~~~~~~~~~~~~~~~~')
      action_dir=self.dataset_dir+('/%s_action' % (action_type))
      files=os.listdir(action_dir)
      files=sorted(files,key=lambda x:int(x.split('.')[0]))
      for i in range(len(files)):
          sys.stdout.write('\rLoad memory>>Load Action_Terminate>>Handling %s '% (files[i]))
          f_action=open(os.path.join(action_dir,files[i]),'r')
          lines=f_action.readlines()
          for j in range(len(lines)):
              if action_type=='coordinate':
                  self.coor_actions[i,j]=int(lines[j])
              elif action_type=='radian':
                  self.radian_actions[i,j]=int(lines[j])
              if j==len(lines)-1:
                  self.terminals[i,j]=1
              else:
                  self.terminals[i,j]=0
                  
  def load_reward(self,reward_type):
     sys.stdout.write('\rLoad memory>>Load Reward from Dataset~~~~~~~~~~~~~~~~~~~')    
     reward_dir=self.dataset_dir+('/%s_reward' % (reward_type))
     files=os.listdir(reward_dir)
     files=sorted(files,key=lambda x:int(x.split('.')[0]))
     for i in range(len(files)):
          sys.stdout.write('\rLoad memory>>Load Reward>>Handling %s '% (files[i]))
          f_reward=open(os.path.join(reward_dir,files[i]),'r')
          lines=f_reward.readlines()
          for j in range(len(lines)):
              if reward_type=='coordinate':
                  self.coor_rewards[i,j]=decimal.Decimal('%.5f' % float(lines[j]))
              else:
                  self.radian_rewards[i,j]=decimal.Decimal('%.5f' % float(lines[j]))
  def load_memory(self):
      self.load_image()
      self.load_action_terminate(action_type='coordinate')
      self.load_action_terminate(action_type='radian')
      self.load_reward(reward_type='coordinate')
      self.load_reward(reward_type='radian')

                  
  
  def get_memorysize(self):
    camera_dirs = os.listdir(self.dataset_dir + '/image')
    camera_dirs=sorted(camera_dirs,key=lambda x:int(x))
    camera=len(os.listdir(camera_dirs))
    squeues=os.listdir(self.dataset_dir + '/image'+'/'+camera_dirs[0])
    squeues=sorted(squeues,key=lambda x:int(x))
    assert camera==self.camera_num("camera dir is not equal camera_num")
    for i in range(camera)-1:
        pre=len(os.path.join(self.dataset_dir + '/image',camera_dirs[i]))
        post = len(os.path.join(self.dataset_dir + '/image', camera_dirs[i+1]))
        assert pre==post ("Camera dirs must have same squeues")
    self.squeues=len(squeues)
    self.image_per_squeues=len(os.listdir(os.path.join(self.dataset_dir + '/image'+'/'+camera_dirs[0],squeues[0])))
    for squeue in squeues:
        for i in range(camera):
            assert self.image_per_squeues==len(os.listdir(os.path.join(self.dataset_dir + '/image'+'/'+camera_dirs[i],squeue)))
    shape=[self.squeues,self.image_per_squeues,camera]
    return shape

      
      
      
      
      