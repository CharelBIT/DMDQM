import os
import shutil
from PIL import Image
import sys
import numpy as np
import decimal
import json
import math
import matplotlib.pyplot as plt
from statistic_analysis import action_statistics


def decimalToAny(num,n):
   baseStr = {10:"a",11:"b",12:"c",13:"d",14:"e",15:"f",16:"g",17:"h",18:"i",19:"j"}
   new_num_str = ""
   if num<n:
       new_num_str=str(num)+new_num_str
       return new_num_str
   while num != 0:
       remainder = num % n
       if 20 > remainder > 9:
           remainder_string = baseStr[remainder]
       elif remainder >=20:
           remainder_string = "("+str(remainder)+")"
       else:
           remainder_string = str(remainder)
       new_num_str = remainder_string+new_num_str
       num = num / n
   return new_num_str


def anyToDecimal(num,n):
   baseStr = {"0":0,"1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,
              "a":10,"b":11,"c":12,"d":13,"e":14,"f":15,"g":16,"h":17,"i":18,"j":19}
   new_num = 0
   nNum = len(num) - 1
   for i in num:        
       new_num = new_num  + baseStr[i]*pow(n,nNum)
       nNum = nNum -1
   return new_num


def save_LineChart(txt_file_path,save_dir):
    fig_name=txt_file_path.split('/')[-1].split('.txt')[0]+'.png'
    f=open(txt_file_path,'r')
    lines=f.readlines()
    distance=[]
    for line in lines:
        distance.append(float(line))
    x=np.linspace(0,len(distance),len(distance))
    
    plt.plot(x,np.array(distance))
    plt.savefig(os.path.join(save_dir,fig_name))
    plt.close()
    

class Preprocess_dataset(object):
    def __init__(self,dataset_dir,dest_dir,sample_dir,is_norm=False):
        self.dataset_dir=dataset_dir
        self.dest_dir=dest_dir
        self.sample_dir=sample_dir
        self.is_norm=is_norm
        
    def make_dirs(self):
        dataset=[self.sample_dir,self.dest_dir]
        dirs=['image','coordinate','radian','coordinate_action','radian_action','info'
            , 'coordinate_distance','radian_distance','dataset_analysis']
        for data in dataset:
            for dir in dirs:
                if not os.path.exists(os.path.join(data, dir)):
                    os.makedirs(os.path.join(data, dir))
        sys.stdout.write('\rChecking the requred dirs if not exits and create ...')

    def form_distance_to_reward(self,distance):
        return 1.0/(10*distance+1)
    
    def line_to_list(self,line):
        #line=line.strip('\n')
        re=[]
        for i in line:
            if i=='':
                continue
            else:
                a=decimal.Decimal('%.4f' % (float(i)))
                re.append(a)
        return re
    def check_image_squeues(self):
        image=os.listdir(self.dataset_dir+'/'+'image')
        image_dir=sorted(image,key=lambda x:int(x))

        location = os.listdir(self.dataset_dir + '/' + 'location')
        location = sorted(image, key=lambda x: int(x))
        if location!=image:
            raise ValueError
    def check_location_image(self):
        loc_path=os.listdir(self.dataset_dir + '/' + 'location')
        loc_dirs=sorted(loc_path,key=lambda x:int(x))
        image_path=os.listdir(self.dataset_dir+'/'+'image')
        image_dirs=sorted(image_path,key=lambda x:int(x))
        for loc_dir,image_dir in zip(loc_dirs,image_dirs):
            assert loc_dir==image_dir('same squeues')
            loc_files=os.listdir(os.path.join(self.dataset_dir+'/'+'location',loc_dir))
            loc_files=sorted(loc_files,key=lambda x:int(x.split('.')[0]))
            image_files=os.listdir(os.path.join(self.dataset_dir+'/'+'image',image_dir))
            image_files=sorted(image_files,key=lambda x:int(x.split('.')[0]))
            for loc_file,image_file in zip(loc_files,image_files):
                assert loc_file.split('.')[0]==image_file.split('.')[0]
                f=open(os.path.join(self.dataset_dir+'/'+'location'+'/'+loc_dir,loc_file),'rb')
                lines=f.readlines()
                try:
                    image=Image.open(os.path.join(self.dataset_dir+'/'+'image'+'/'+image_dir,image_file))
                    image.verify()

                except:

                    sys.stdout.write('\rImage can\'t be read %s squeues' % (image_dir))
                    shutil.rmtree(self.dataset_dir+'/'+'location'+'/'+loc_dir)
                    shutil.rmtree(self.dataset_dir+'/'+'image'+'/'+image_dir)
                    break
                if len(lines)>1:
                    sys.stdout.write('\rchecking location and remove %s squeues' % (loc_dir))
                    shutil.rmtree(self.dataset_dir+'/'+'location'+'/'+loc_dir)
                    shutil.rmtree(self.dataset_dir+'/'+'image'+'/'+image_dir)
                    break
    def check_dataset(self):
        self.check_image_squeues()
        self.check_location_image()
                            
    def euclideandistance(self,line1,line2):
        assert len(line1)==len(line2)
        distance=0.0
        for i in range(len(line1)):
            distance=math.pow((line1[i]-line2[i]),2)+distance
        return math.sqrt(distance)
      
    def copy_processing(self):
        sys.stdout.write('\r>>~~~~~~~~~~~~~~~~~Copy Preocessing ~~~~~~~~~~~~~~~~~~~~~~~~~')
        image_date_dirs=[]
        
        date_dirs=os.listdir(self.dataset_dir)
        data_dirs=sorted(date_dirs,key=lambda x:int(x))
        for i in data_dirs:
            i_dir=os.path.join(self.dataset_dir,i)
            if os.path.isdir(i_dir):
                image_date_dirs.append(os.path.join(os.path.join(self.dataset_dir,i)))
        num_squeue=0
        for image_date_dir in image_date_dirs:
            
            image_squeues=os.listdir(image_date_dir)
            image_squeues=sorted(image_squeues,key=lambda x:int(x))
            for image_squeue in image_squeues:
                for file_name in os.listdir(os.path.join(image_date_dir,image_squeue)):
                    if file_name.endswith('.jpg'):
                        if not os.path.exists(os.path.join(self.dest_dir+'/image',str(num_squeue))):
                            os.makedirs(os.path.join(self.dest_dir+'/image',str(num_squeue)))
                    
                        shutil.copy(os.path.join(image_date_dir,image_squeue+'/'+file_name),
                                    os.path.join(self.dest_dir,'image/'+str(num_squeue)+'/'+file_name))
                    if file_name.endswith('.txt'):
                        if not os.path.exists(os.path.join(self.dest_dir+'/info',str(num_squeue))):
                                os.makedirs(os.path.join(self.dest_dir+'/info',str(num_squeue)))
                        shutil.copy(os.path.join(image_date_dir,image_squeue+'/'+file_name),
                                       os.path.join(self.dest_dir,'info/'+str(num_squeue)+'/'+file_name))
                        
                num_squeue=num_squeue+1
                
                sys.stdout.write("\r Copy Preocessing > > Handling %s squeues" % (image_squeue))
                sys.stdout.flush()  
                
    def gen_radian_and_coordinate(self,dest_dir=None):
        sys.stdout.write("\r>>~~~~~~~~~~~~~~Generating coordinates and radian~~~~~~~~~~~~~~~~~~")
        sys.stdout.flush()
        info_dir=self.dest_dir+'/info'
        coordinate_dir=self.dest_dir+'/coordinate'
        radian_dir=self.dest_dir+'/radian'
        squeues=os.listdir(info_dir)
        squeues=sorted(squeues,key=lambda x:int(x))
        for squeue in squeues:
            f_coor=open(os.path.join(coordinate_dir,squeue+'.txt'),'w')
            f_radian=open(os.path.join(radian_dir,squeue+'.txt'),'w')
            info_files_path=os.path.join(info_dir,squeue)
            info_files=os.listdir(info_files_path)
            info_files=sorted(info_files,key=lambda x:int(x.split('.')[0]))
            for info_file in info_files:
                f_info=open(os.path.join(info_files_path,info_file),'r')
                lines=f_info.readlines()
                for line in lines:
                    line_coors=line.split(',')[:3]
                    line_radians=line.split(',')[3:6]
                    f_coor.write(''.join(line_coors)+'\n')
                    for i in range(len(line_radians)):
                        if i ==0:
                            line_radians[i]=line_radians[i]+' '
                        elif i==1:
                            line_radians[i]=line_radians[i]
                        elif i==2:
                            line_radians[i]=line_radians[i].split('\t')[0]+' '
                    f_radian.write(''.join(line_radians)+'\n')
                    
                f_info.close()
            f_coor.close()
            f_radian.close()
            sys.stdout.write('\rGenerating coordinates > > Handling %s squeues' % (squeue.split('.'[0])))
                
    def gen_coordinate_radian_action(self,dest_dir=None):
        if dest_dir==None:
            coordinate_path=self.dest_dir+'/'+'coordinate'
            coordinate_action_path=self.dest_dir+'/'+'coordinate_action'
            
            radian_path=self.dest_dir+'/'+'radian'
            radian_action_path=self.dest_dir+'/'+'radian_action'
            sys.stdout.write("\r>>~~~~~~~~~~~~~~Generating coordinate Actions and radian Actions~~~~~~~~~~~~~~~~~~")
        else:
            sys.stdout.write("\r>>Sampling > > ~~~~~~~Generationg Actions and randian Actions~~~~~~~~~~~~~~~~~~")
            coordinate_path=dest_dir+'/'+'coordinate'
            coordinate_action_path=dest_dir+'/'+'coordinate_action'
            
            radian_path=dest_dir+'/'+'radian'
            radian_action_path=dest_dir+'/'+'radian_action' 
            
        file_names=os.listdir(coordinate_path)
        file_names=sorted(file_names,key=lambda x:int(x.split('.')[0]))          
        for file_name in file_names:
            f_coor=open(os.path.join(coordinate_path,file_name),'r')
            f_coor_action=open(os.path.join(coordinate_action_path,file_name),'w')
    
            f_rad=open(os.path.join(radian_path,file_name),'r')
            f_rad_action=open(os.path.join(radian_action_path,file_name),'w')
            
            
            lines_coor=f_coor.readlines()
            lines_rad=f_rad.readlines()
            assert len(lines_coor)==len(lines_rad)
            for i in range(len(lines_coor)-1):
                pre_line_coor=lines_coor[i].strip('\n').split(' ')
                post_line_coor=lines_coor[i+1].strip('\n').split(' ')

                pre_line_rad=lines_rad[i].strip('\n').split(' ')
                post_line_rad=lines_rad[i+1].strip('\n').split(' ')
                
                assert len(pre_line_coor)==len(post_line_coor)
                assert len(pre_line_rad)==len(post_line_rad)
                assert len(pre_line_coor)==len(post_line_rad)
                
                result_coor=self.comp_pre_post(pre_line_coor, post_line_coor)
                result_rad=self.comp_pre_post(pre_line_rad, post_line_rad)

                coor_action=anyToDecimal(result_coor, 3)  
                rad_action=anyToDecimal(result_rad, 3)     
                f_coor_action.write(str(coor_action)+'\n')  
                f_rad_action.write(str(rad_action)+'\n')  
                
            f_coor_action.write(str(13)+'\n')
            f_rad_action.write(str(13)+'\n')
            
            f_coor.close()
            f_rad.close()
            
            f_coor_action.close()
            f_rad_action.close()
            
            if dest_dir==None:    
                sys.stdout.write('\rGenerating Actions > > Handling %s squeueus' % (file))
                sys.stdout.flush()
            else:
                sys.stdout.write('\r>>Sampling > > Generating Actions> > Handling %s squeueus' % (file))
                sys.stdout.flush()  
                               
    def comp_pre_post(self,pre,post):
        result=[]
        assert len(pre)==len(post)
        for i in range(len(pre)-1):
            pre_pos=decimal.Decimal('%.4f' % (float(pre[i])))
            post_pos=decimal.Decimal('%.4f' % (float(post[i])))
            if post_pos-pre_pos<0:
                result.append('0')
            elif post_pos-pre_pos==0:
                result.append('1')
            elif post_pos-pre_pos>0:
                result.append('2')
        return result
        
                   
                
    def overlook(self):
        over_all={}
        num_pict=[]
        
        info_path=self.dest_dir+'/info/dataset_analysis'
        info_dirs=os.listdir(info_path)
        info_dirs=sorted(info_dirs,key=lambda x :int(x))
        for info_dir in (info_dirs):
            file=os.listdir(os.path.join(info_path,info_dir))
            num_pict.append([int(info_dir),len(file)])
        over_all['Image_Number']=num_pict
        over_all['Max_Number']=max(np.array(num_pict)[:,1])
        over_all['Min_Number']=min(np.array(num_pict)[:,1])
        over_all['Aver_Number']=sum(np.array(num_pict)[:,1])*1.0/len(num_pict)
        file_object=json.dumps(over_all)
        file_json=open(os.path.join(self.dest_dir,'overlook.json'),'w')
        file_json.write(file_object)
        file_json.close()
        
    def gen_distance(self,dest_dir=None,distance_tyep='coordinate'):
        sys.stdout.write('\r~~~~~~~~Grenerating Distance~~~~~~~~~~~~~')
        if dest_dir==None:
            coordinate_path=self.dest_dir+'/'+distance_tyep
            distance_path=self.dest_dir+('/%s_distance' % (distance_tyep))
        else:
            coordinate_path=dest_dir+'/'+distance_tyep
            distance_path=dest_dir+('/%s_distance' % (distance_tyep))         
        files=os.listdir(coordinate_path)
        files=sorted(files,key=lambda x:int(x.split('.')[0]))
        for file in files:
            f=open(os.path.join(coordinate_path,file),'r')
            f_distace=open(os.path.join(distance_path,file),'w')
            lines=f.readlines()
            last_line_to_list=self.line_to_list(lines[len(lines)-1].strip('\n').split(' '))
            for line in lines:
                now_line_to_list=self.line_to_list(line.strip('\n').split(' '))
                distance=self.euclideandistance(now_line_to_list, last_line_to_list)
                distance=decimal.Decimal('%.4f' % (distance))
                f_distace.write(str(distance)+'\n')
            if dest_dir==None:    
                sys.stdout.write('\rGrenerating Distance > >Handling %s squeueus' % (file))
                sys.stdout.flush()
            else:
                sys.stdout.write('\r>>Sampling > > Handling %s squeueus' % (file))
                sys.stdout.flush()            
    def analysis_action(self,dest_dir=None,action_type='coordinate',action_overturn=True):
        if dest_dir==None:
            
            action_path=self.dest_dir+('/%s_action' % (action_type))
            action_plot=self.dest_dir+'/dataset_analysis/'+action_type
        else:
            action_path=dest_dir+('/%s_action' % (action_type))
            action_plot=dest_dir+'/dataset_analysis/'+action_type            
        action_analysis=action_statistics(action_path,27,action_plot,action_type,action_overturn)
        action_analysis.load_actions()
        action_analysis.save_hist()
        
    def analysis_distance(self,dest_dir=None,distance_type='coordinate'):
        sys.stdout.write('\r>>~~~~~~Analysis distance~~~~~~~~~~~~')
        sys.stdout.flush()
        if dest_dir==None:
            analysis_distace_dir=self.dest_dir+('/dataset_analysis/%s_distacnce' % (distance_type))
            distance_dir=self.dest_dir+('/%s_distance' %(distance_type))
        else:
            analysis_distace_dir=dest_dir+('/dataset_analysis/%s_distacnce' % (distance_type))
            distance_dir=dest_dir+('/%s_distance' % (distance_type))
        if not os.path.exists(analysis_distace_dir):
            os.makedirs(analysis_distace_dir)
        files=os.listdir(distance_dir)
        files=sorted(files,key=lambda x:int(x[:-4]))
        for file in files:
            sys.stdout.write('\r>>Handling %s' % (file))
            sys.stdout.flush()
            save_LineChart(os.path.join(distance_dir,file),analysis_distace_dir)
            
    def analysis_reward(self,dest_dir=None,reward_type='coordinate'):
        sys.stdout.write('\r>>~~~~~~Analysis Reward~~~~~~~~~~~~')
        sys.stdout.flush()
        if dest_dir==None:
            analysis_distace_dir=self.dest_dir+'/dataset_analysis/reward'
            distance_dir=self.dest_dir+'/reward'
        else:
            analysis_distace_dir=dest_dir+'/dataset_analysis/reward'
            distance_dir=dest_dir+('/%s_reward' % (reward_type))
        if not os.path.exists(analysis_distace_dir):
            os.makedirs(analysis_distace_dir)
        files=os.listdir(distance_dir)
        files=sorted(files,key=lambda x:int(x[:-4]))
        for file in files:
            sys.stdout.write('\r>>Handling %s' % (file))
            sys.stdout.flush()
            save_LineChart(os.path.join(distance_dir,file),analysis_distace_dir)        
            
                
    def sampling_image(self,squeues_ind): 
        source_dir=self.dest_dir+'/image'
        dst_dir=self.sample_dir+'/image'
        
        assert squeues_ind.shape[0]==len(os.listdir(source_dir))
        
        source_paths=os.listdir(source_dir)
        source_paths=sorted(source_paths,key=lambda x:int(x))
        
        for i in range(len(source_paths)):
            if not os.path.exists(os.path.join(dst_dir,source_paths[i])):
                os.makedirs(os.path.join(dst_dir,source_paths[i]))
            files=os.listdir(os.path.join(source_dir,source_paths[i]))
            files=sorted(files,key=lambda x:int(x.split('.')[0]))
            num_sample=0
            inds_list=list(squeues_ind[i,:])
            for ind in inds_list:
                shutil.copy(os.path.join(source_dir,source_paths[i]+'/'+files[ind]),
                            os.path.join(dst_dir,source_paths[i]+'/'+str(num_sample)+'.jpg'))
                num_sample=num_sample+1


    def sampling_radian_coordinate(self,squeues_ind): 
        coor_source_dir=self.dest_dir+'/coordinate'
        coor_dst_dir=self.sample_dir+'/coordinate'
        
        radian_source_dir=self.dest_dir+'/radian'
        radian_dst_dir=self.sample_dir+'/radian'
        
        assert squeues_ind.shape[0]==len(os.listdir(coor_source_dir))
        assert squeues_ind.shape[0]==len(os.listdir(radian_source_dir))
        
        txt_files=os.listdir(coor_source_dir)
        txt_files=sorted(txt_files,key=lambda x:int(x.split('.')[0]))
        
        for i in range(len(txt_files)):
            f_coor_source=open(os.path.join(coor_source_dir,txt_files[i]),'r')
            f_coor_dst=open(os.path.join(coor_dst_dir,txt_files[i]),'w')

            f_radian_source=open(os.path.join(radian_source_dir,txt_files[i]),'r')
            f_radian_dst=open(os.path.join(radian_dst_dir,txt_files[i]),'w')
            lines_coor=f_coor_source.readlines()
            lines_radian=f_radian_source.readlines()
            inds=list(squeues_ind[i,:])
            for ind in inds:
                f_coor_dst.write(lines_coor[ind])
                f_radian_dst.write(lines_radian[ind])
                
            f_coor_source.close()
            f_radian_source.close()
            
            f_coor_dst.close()
            f_radian_dst.close()
                
    def sample_policy(self,start_index=50,num_sample=60,policy='linear'):
        num_every_squeue=[]
        inds=sorted(range(num_sample),reverse=True)
        squeues_dir=self.dest_dir+'/image'
        squeues=os.listdir(squeues_dir)
        squeues=sorted(squeues,key=lambda x:int(x))
        squeues_ind=np.zeros((len(squeues),num_sample),dtype=np.int)
        for squeue in squeues:
            number=len(os.listdir(os.path.join(squeues_dir,squeue)))-1
            num_every_squeue.append(number)
        if policy=='linear':
            for squeue in range(len(num_every_squeue)):
                space=(num_every_squeue[squeue]-51)/num_sample
                for ind in inds:
                    squeues_ind[squeue,ind]=num_every_squeue[squeue]-1-(num_sample-1-ind)*space
        return squeues_ind
    
    def sampling(self,start_index=50,num_sample=60,policy='linear'):
        squeues_ind=self.sample_policy(start_index, num_sample, policy)
        self.sampling_image(squeues_ind)
        self.sampling_radian_coordinate(squeues_ind)
        self.gen_coordinate_radian_action(self.sample_dir)   
        
        
    def gen_reward(self,dest_dir=None,reward_type='coordinate'):
        sys.stdout.write('\rGenerating Reward~~~~~~~~~~~')
        if dest_dir==None:
            distance_dir=self.dest_dir+('/%s_distance' % (reward_type))
            reward_dir=self.dest_dir+('/%s_reward' % (reward_type))
        else:
            distance_dir=dest_dir+('/%s_distance' % (reward_type))
            reward_dir=dest_dir+('/%s_reward' % (reward_type))
        files=os.listdir(distance_dir)
        files=sorted(files,key=lambda x:int(x[:-4]))
        for file in files:

            f_dist=open(os.path.join(distance_dir,file),'r')
            f_reward=open(os.path.join(reward_dir,file),'w')
            lines=f_dist.readlines()
            for i in range(len(lines)):
                if self.is_norm==False:
                    distance=float(lines[i])
                    reward=self.form_distance_to_reward(distance)
                else:
                    if i<len(lines)-1:
                        reward=np.sign(float(lines[i]-lines[i+1]))
                    else:
                        reward=0.0
                f_reward.write(str(reward)+'\n')
            sys.stdout.write('\rGenerating Reward > > Handling %s' % (file))
            sys.stdout.flush()                
        
                 
            
                   
a=Preprocess_dataset('/home/officer/data/0406','/home/officer/data/dataset','/home/officer/data/dataset_sample')
# 
# #a.overlook()
# a.make_dirs()
# a.remove_squeue()
# a.copy_processing()
# a.gen_radian_and_coordinate()
# a.gen_coordinate_radian_action()
# a.gen_distance()
# a.gen_distance(distance_tyep='radian')
#  
# a.analysis_action(action_type='radian')
# a.analysis_action(action_type='coordinate')
# a.analysis_distance()
# a.analysis_distance(distance_type='radian')
#  
# a.sampling()
# a.gen_coordinate_radian_action(dest_dir='/home/officer/data/dataset_sample')
# a.gen_distance(dest_dir='/home/officer/data/dataset_sample')
# a.gen_distance(dest_dir='/home/officer/data/dataset_sample',distance_tyep='radian')
# a.analysis_action(dest_dir='/home/officer/data/dataset_sample',action_type='radian')
# a.analysis_action(dest_dir='/home/officer/data/dataset_sample',action_type='coordinate')
# a.analysis_distance(dest_dir='/home/officer/data/dataset_sample')
# a.analysis_distance(dest_dir='/home/officer/data/dataset_sample',distance_type='radian')
# a.gen_reward(dest_dir='/home/officer/data/dataset_sample',reward_type='coordinate')
# a.gen_reward(dest_dir='/home/officer/data/dataset_sample',reward_type='radian')
# a.analysis_reward(dest_dir='/home/officer/data/dataset_sample',reward_type='coordinate')
# a.analysis_reward(dest_dir='/home/officer/data/dataset_sample',reward_type='radian')

