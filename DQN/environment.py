import random
import numpy as np
from .utils import rgb2gray, imresize

import socket
import time
import struct
import math
class Environment(object):
  def __init__(self, config):
    screen_width, screen_height, self.action_repeat, self.random_start = \
        config.screen_width, config.screen_height, config.action_repeat, config.random_start

    self.display = config.display
    self.dims = (screen_width, screen_height)

    self._screen = None
    self.reward = 0
    self.terminal = True
    self.action_size=config.action_size
    self.is_train=True
    
class ARMEnv(object):
    def __init__(self,config):
        screen_width, screen_height, self.action_repeat, self.random_start = \
        config.screen_width, config.screen_height, config.action_repeat, config.random_start

        self.display = config.display
        self.dims = (screen_width, screen_height)

        self._screen = None
        self.reward = 0
        self.terminal = True
        self.coor_action_size=config.coor_action_size
        self.radian_action_size=config.radian_action_size
        self.is_train=True
        self.HOST=config.HOST
        self.PORT=config.PORT
        self.radian_move_step=config.radian_move_step
        self.coor_move_step=config.coor_move_step
        self.s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#      self.s.connect((self.HOST,self.PORT))
        time.sleep(1.00)
        self.s.close()
    def getPose(self):
        print "The current position:"
        self.s.close()
        self.s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.HOST,self.PORT))
        time.sleep(1.00)
        self.packet_1 = self.s.recv(4)
        self.packet_2 = self.s.recv(8)
        self.packet_3 = self.s.recv(48)
        self.packet_4 = self.s.recv(48)
        self.packet_5 = self.s.recv(48)
        self.packet_6 = self.s.recv(48)
        self.packet_7 = self.s.recv(48)
        self.packet_8 = self.s.recv(48)
        self.packet_9 = self.s.recv(48)
        self.packet_10 = self.s.recv(48)
        self.packet_11 = self.s.recv(48)
        #get_x_pose
        self.packet_12 = self.s.recv(8)
        self.packet_12 = self.packet_12.encode("hex")
        self.x = str(self.packet_12)
        self.x = struct.unpack('!d', self.packet_12.decode('hex'))[0]
        print "X = ", self.x*1000
        #get_y_pose
        self.packet_13 = self.s.recv(8)
        self.packet_13 = self.packet_13.encode("hex")
        self.y = str(self.packet_13)
        self.y = struct.unpack('!d', self.packet_13.decode('hex'))[0]
        print "Y = ", self.y*1000
        #get_z_pose
        self.packet_14 = self.s.recv(8)
        self.packet_14 = self.packet_14.encode("hex")
        self.z = str(self.packet_14)
        self.z = struct.unpack('!d', self.packet_14.decode('hex'))[0]
        print "Z = ", self.z*1000
        #get_Rx_rota
        self.packet_15 = self.s.recv(8)
        self.packet_15 = self.packet_15.encode("hex")
        self.Rx = str(self.packet_15)
        self.Rx = struct.unpack('!d', self.packet_15.decode('hex'))[0]
        print "Rx = ", self.Rx
        #get_Ry_rota
        self.packet_16 = self.s.recv(8)
        self.packet_16 = self.packet_16.encode("hex")
        self.Ry = str(self.packet_16)
        self.Ry = struct.unpack('!d', self.packet_16.decode('hex'))[0]
        print "Ry = ", self.Ry
        #get_Rz_rota
        self.packet_17 = self.s.recv(8)
        self.packet_17 = self.packet_17.encode("hex")
        self.Rz = str(self.packet_17)
        self.Rz = struct.unpack('!d', self.packet_17.decode('hex'))[0]
        print "Rz = ", self.Rz
        return [self.x*1000,self.y*1000,self.z*1000],[self.Rx,self.Ry,self.Rz]


    def move(self,coor_result,radian_result):
        self.x=self.x+(int(coor_result[0])-1)*self.coor_move_step
        self.y=self.y+(int(coor_result[1])-1)*self.coor_move_step
        self.z=self.z+(int(coor_result[2])-1)*self.coor_move_step
        
        self.Rx=self.Rx+(int(radian_result[0])-1)*self.radian_move_step
        self.Ry=self.Ry+(int(radian_result[1])-1)*self.radian_move_step
        self.Rz=self.Rz+(int(radian_result[2])-1)*self.radian_move_step
        
        self.deltax_=str(self.x)
        self.deltay_=str(self.y)
        self.deltaz_=str(self.z)
        
        self.deltaRx_=str(self.Rx)
        self.deltaRy_=str(self.Ry)
        self.deltaRz_=str(self.Rz)
        self.s.close()
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.HOST,self.PORT))
        # movej(q, a=1.4, v=1.05, t=0, r=0)
        self.s.send("movej(get_inverse_kin(p["+self.deltax_+","+self.deltay_+","+self.deltaz_+","+self.deltaRx_+","+self.deltaRy_+","+self.deltaRz_+"]),a=0.5, v=0.5, t=0, r=0)\r\n")
        # self.s.send("servoj(get_inverse_kin(p["+self.deltax_+","+self.deltay_+","+self.deltaz_+","+self.deltaRx_+","+self.deltaRy_+","+self.deltaRz_+"]),t=5,lookahead_time=0.3,gain=300)\r\n")
        # print "servoj(get_inverse_kin(p["+self.deltax_+","+self.deltay_+","+self.deltaz_+","+self.deltaRx_+","+self.deltaRy_+","+self.deltaRz_+"]),t=0.5,lookahead_time=0.3,gain=300)\r\n"
           
def PtInAnyRect(p,RectPoint):
    nCount=4
    nCross=0
    for i in range(nCount):
        pStart=RectPoint[i]
        print pStart
        pEnd=RectPoint[(i+1)%nCount]
        if pStart[1]==pEnd[1]:
            continue
        if p[1]< min(pStart[1],pEnd[1]) or  p[1]>max(pStart[1],pEnd[1]):
            continue
        d = (p[1]-pStart[1])*(pEnd[0]-pStart[0])/(pEnd[1]-pStart[1])+pStart[0]
        if d>p[0]:
            nCross = nCross+1
    return (nCross%2==1)
        


    def init_position(self):
        pose_selected = np.array([[-646.7,-162.1,41.4],[-505.2,-8.1,-73.4],[-348.5,-266.1,-91.8],[-457.9,-490.1,41.4]])

        flag=True
        while(flag):
            x = random.uniform(-646,-348)
            y = random.uniform(-490,-8)
            z = random.uniform(350,440)
            b = np.array([[x,y]])
            if PtInAnyRect(b[0],pose_selected):
                break


        x0=x/1000
        y0=y/1000
        z0=z/1000
        Rx0=1.9
        Ry0=0.9
        Rz0=-1.7
        self.x=x0
        self.y=y0
        self.z=z0
        self.Rx=Rx0
        self.Ry=Ry0
        self.Rz=Rz0
        self.deltax_=str(self.x)
        self.deltay_=str(self.y)
        self.deltaz_=str(self.z)
        self.deltaRx_=str(self.Rx)
        self.deltaRy_=str(self.Ry)
        self.deltaRz_=str(self.Rz)
        self.s.close()
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.HOST,self.PORT))
        self.s.send("movej(get_inverse_kin(p["+self.deltax_+","+self.deltay_+","+self.deltaz_+","+self.deltaRx_+","+self.deltaRy_+","+self.deltaRz_+"]),a=0.5, v=0.5, t=0, r=0)\r\n")













    
