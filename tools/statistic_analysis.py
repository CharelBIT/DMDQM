import os
import numpy as np
import json
from matplotlib import mlab
import matplotlib.pyplot as plt
import decimal


class action_statistics(object):
    def __init__(self,action_dir,action_num,save_hist_dir,action_type,action_overturn=False):
        self.action_type=action_type
        self.action_dir=action_dir
        self.action_files=self.dir_list()
        self.action_num=action_num
        self.action=np.zeros((action_num),dtype=np.int32)
        self.save_hist_dir=save_hist_dir
        self.action_overturn=action_overturn
        
    def load_actions(self):
        for action_file in self.action_files:
            self.load_per_file(action_file)
        
        
    def dir_list(self):
        return [actions for actions in os.listdir(self.action_dir) if actions.endswith('.txt') ]
    
    def load_per_file(self,txt_file):
        f=open(os.path.join(self.action_dir,txt_file),'r')
        lines=f.readlines()
        for line in lines:
            self.action[int(line)]=self.action[int(line)]+1
            if self.action_overturn:
                self.action[self.action_num-1-int(line)]=self.action[self.action_num-1-int(line)]+1
            
    def save_hist(self):
#        fig1=plt.figure()
        plt.bar(range(0,self.action_num),self.action)
        plt.title(self.action_type)
        plt.savefig(self.save_hist_dir+'%s_action_hist.png' %(self.action_type))
        plt.close()
        
            
    def draw_hist(self):
        fig1=plt.figure(1)
        rects=plt.bar(range(0,self.action_num),self.action)
        plt.title('action_hist')
        plt.show()
        
        
def line_to_list(self,line):
        #line=line.strip('\n')
    re=[]
    for i in line:
        if i=='':
            continue
        else:
            a=decimal.Decimal('%.3f' % (float(i)))
            re.append(a)
    return re
        
def draw_LineChart(txt_file_dir):
    f=open(txt_file_dir,'r')
    lines=f.readlines()
    distance=[]
    for line in lines:
        distance.append(float(line))
    x=np.linspace(0,len(distance),len(distance))
    plt.plot(x,np.array(distance))
    plt.show()
               
        