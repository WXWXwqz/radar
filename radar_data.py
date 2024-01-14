import datetime
import bisect
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
from typing import List
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score
import pandas as pd
import time
from multiprocessing import Process
from collections import Counter
import csv
def get_npy_dataset(dateset_list):
    # dateset_list = ['acc.npy', 'normal.npy','acc1.npy','normal1.npy','normal2.npy']
    x_data = None
    for datename in dateset_list:
        X_train = np.load(datename)
        if 'acc' in datename:
            y_train = np.ones(X_train.shape[0])
        elif 'normal' in datename:
            y_train = np.zeros(X_train.shape[0])
        elif 'inference' in datename:
            y_train = np.ones(X_train.shape[0])+1
        print(X_train.shape)
        if x_data is None:
            x_data = X_train
            y_data = y_train
        else:
            x_data = np.concatenate((x_data, X_train), axis=0)
            y_data = np.concatenate((y_data, y_train), axis=0)
    return x_data,y_data
def get_npy_feature(start_time,end_time,save_names,dir,path='./data/mmAcc/5008/pkl/'):
    radar_feature = Radar_Feature(start_time=start_time,end_time=end_time,path=path)
    radar_feature.cal_feature([dir])
    np.save(save_names,radar_feature.feature_data_dict[dir])
def find_closest_numbers(nums):
    closest_nums = []
    for i, num in enumerate(nums):
        closest = None
        closest_distance = float('inf')
        for j, other_num in enumerate(nums):
            if i != j:
                distance = abs(num - other_num)
                if distance < closest_distance:
                    closest_distance = distance
                    closest = other_num
        closest_nums.append(closest)
    return closest_nums

class Lane_Info:
    def __init__(self,frame) -> None:
        self.no = frame[0]
        self.color = frame[1]
        self.cd_time = int.from_bytes(frame[2:4],byteorder='little')
        self.byLaneQueueLength = frame[4]
        self.lon_last_car = int.from_bytes(frame[5:9],byteorder='little')
        self.lat_last_car = int.from_bytes(frame[9:13],byteorder='little')
        self.QueueVehNum = frame[13]
        self.SectionVehNum = frame[14]
        self.SpaceOccupancy = frame[15]
        self.AveSpeed = frame[16]
        self.HeadLocation = frame[17]
        self.HeadSpeed = frame[18]

class Obj_Info:    
    def __init__(self,frame) -> None:
        # print(len(frame))
        self.obj_id = int.from_bytes(frame[0:2],byteorder="little")
        self.obj_len = frame[2]
        self.obj_width = frame[3]
        self.yaw = int.from_bytes(frame[4:6],byteorder='little')
        self.x = int.from_bytes(frame[6:10],byteorder='little',signed=True) # 0.01m
        self.y = int.from_bytes(frame[10:14],byteorder='little',signed=True) #0.01m
        self.speed = int.from_bytes(frame[14:16],byteorder='little') # 0.01km/h
        self.radar_dir = frame[16]
        self.car_num = frame[17:37]
        self.lon = int.from_bytes(frame[37:41],byteorder='little')
        self.lat = int.from_bytes(frame[41:45],byteorder='little')
        self.obj_type = frame[45]
        self.car_color = frame[46]
        self.plate_color = frame[47]
        self.stop_num = frame[48]
        self.paking_delay = int.from_bytes(frame[49:51],byteorder='little')
        self.obj_status = frame[51]
        self.is_in_lane = frame[52]  #1 进口�??? 0 出口�???
        self.track_source = frame[53]
        self.video_dir = frame[54]
        self.video_id = frame[55]
        self.video_car_typd = frame[56]
        self.video_obj_id = int.from_bytes(frame[57:59],byteorder='little')
        self.radar_obj_id = int.from_bytes(frame[59:61],byteorder='little')
        self.obj_flux = frame[61]   #目标转向
        self.obj_lanenum = frame[62]  #车道�??
        self.RCS = int.from_bytes(frame[63:65],byteorder='little')
        self.vX = int.from_bytes(frame[65:67],byteorder='little',signed=True)
        self.vY = int.from_bytes(frame[67:69],byteorder='little',signed=True)
        self.accX = int.from_bytes(frame[69:71],byteorder='little',signed=True)
        self.axxY = int.from_bytes(frame[71:73],byteorder='little',signed=True)
        
        None

class Radar_Feature:
    def __init__(self,start_time:datetime.datetime,end_time:datetime.datetime,path) -> None:
        self.start_time=start_time
        self.end_time = end_time
        # self.feature_dim = len(feature_names)
        
        self.feature_func_dict={}
        self.feature_func_dict['obj_num']=self.cal_feature_obj_num
        # self.feature_func_dict['obj_num_rate']=self.cal_feature_obj_num_rate
        self.feature_func_dict['obj_num_highspeed']=self.cal_feature_obj_num_high_speed
        # self.feature_func_dict['obj_num_highspeed_rate']=self.cal_feature_obj_num_high_speed_rate
        self.feature_func_dict['obj_num_lowspeed']=self.cal_feature_obj_num_low_speed
        # self.feature_func_dict['obj_num_lowspeed_rate']=self.cal_feature_obj_num_low_speed_rate
        self.feature_func_dict['obj_avespeed']=self.cal_feature_ave_speed
        self.feature_func_dict['obj_aveaccX']=self.cal_feature_ave_accX
        self.feature_func_dict['obj_aveaccY']=self.cal_feature_ave_accY       

        self.feature_func_dict['obj_avexspeed']=self.cal_feature_ave_x_speed
        self.feature_func_dict['obj_aveyspeed']=self.cal_feature_ave_y_speed
        # self.feature_func_dict['obj_avespeed_rate']=self.cal_feature_ave_speed_rate
        # self.feature_func_dict['obj_avexspeed_rate']=self.cal_feature_ave_x_speed_rate
        # self.feature_func_dict['obj_aveyspeed_rate']=self.cal_feature_ave_y_speed_rate

        self.feature_func_dict['obj_speed_std']=self.cal_feature_speed_std
        self.feature_func_dict['obj_xspeed_std']=self.cal_feature_x_speed_std
        self.feature_func_dict['obj_yspeed_std']=self.cal_feature_y_speed_std        

        self.feature_func_dict['obj_avg_nearest_X_dist']=self.cal_feature_avg_nearest_X_dist
        self.feature_func_dict['obj_avg_nearest_Y_dist']=self.cal_feature_avg_nearest_Y_dist
        self.feature_func_dict['obj_avg_nearest_XY_dist']=self.cal_feature_avg_nearest_XY_dist
        self.feature_func_dict['lane_speed_std']=self.cal_feature_lane_speed_std
        self.feature_func_dict['lane_xspeed_std']=self.cal_feature_lane_xspeed_std
        self.feature_func_dict['lane_yspeed_std']=self.cal_feature_lane_yspeed_std

        self.feature_func_dict['lane_objnum_std']=self.cal_feature_lane_objnum_std
        self.feature_func_dict['is_lane_in']=self.cal_featrue_lane_is_in
        # self.feature_data_dict['lane_color']=self.cal_feature_lane_color

        
        self.feature_name = list(self.feature_func_dict.keys())[:]
        self.speed_th = 1000
        self.frame_diff_N = 1000

        


        self.feature_dim = len(self.feature_name)
        self.feature_seque_len = 300  # 100ms 一个数据，一帧数据的时间为feature_seque_len*0.1s
        self.feature_seque_offset = 5 # 1s 间隔进行数据偏移
        
        self.raw_data_file_name=[]
        start_timet = time.time()  
        self.raw_data:List[Radar_Dat] = self.get_raw_data(path)
        end_timet = time.time()  
        execution_time = end_timet - start_timet          
        # print(f"执行时间001: {execution_time} 秒")
        self.raw_data_time = [item.time for item in self.raw_data]
        self.feature_time_dict={}
        self.feature_data_dict={}
        self.index = bisect.bisect(self.raw_data_time, self.start_time)
        self.indexend = bisect.bisect(self.raw_data_time, self.end_time)
        # self.cal_feature([11,31,51,71])
        None
    
    def cal_feature_lane_color(self,s_index,dir):

        lane_color_set={}
        lane_color_set[1]=[1,2,3,4,5,22,23,24]
        lane_color_set[2]=[6,7,8,14,17,18,19,20,21]
        lane_light_color_set={}
        re= []
        seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]        
        for d in seque_data:
            for lane in d.lane_info:

                for k,v in lane_color_set.items():

                    if lane.no in v:
                        if k not in lane_light_color_set.keys():
                            lane_light_color_set[k]=[lane.color]
                        else:
                            lane_light_color_set[k].append(lane.color)

                

                if lane.no==dir:
                    re.append(lane.color)        

    def cal_featrue_lane_is_in(self,s_index,dir):
        re= []
        seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]        
        for d in seque_data:
            re.append(dir%10)    
        return np.array(re).reshape(len(re),1)
    
    def cal_feature_ave_speed(self,s_index,dir): 
        re= []
        seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]        
        for d in seque_data:
            obj_num =0 
            obj_dir = [obj.speed/100 for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)] #速度转换�????1km/h 或�?1m/s
            if len(obj_dir)==0:
                re.append(0)
            else:
                re.append(sum(obj_dir)/len(obj_dir))        
        return np.array(re).reshape(len(re),1)
    def cal_feature_ave_accX(self,s_index,dir): 
        re= []
        seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]        
        for d in seque_data:
            obj_num =0 
            obj_dir = [obj.accX/100 for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)] #速度转换�????1km/h 或�?1m/s
            if len(obj_dir)==0:
                re.append(0)
            else:
                re.append(sum(obj_dir)/len(obj_dir))        
        return np.array(re).reshape(len(re),1)
    def cal_feature_ave_accY(self,s_index,dir): 
        re= []
        seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]        
        for d in seque_data:
            obj_num =0 
            obj_dir = [obj.accX/100 for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)] #速度转换�????1km/h 或�?1m/s
            if len(obj_dir)==0:
                re.append(0)
            else:
                re.append(sum(obj_dir)/len(obj_dir))        
        return np.array(re).reshape(len(re),1)
    def cal_feature_ave_x_speed(self,s_index,dir): 
        re= []
        seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]        
        for d in seque_data:
            obj_num =0 
            obj_dir = [obj.vX/100 for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)] #速度转换�????1km/h 或�?1m/s
            if len(obj_dir)==0:
                re.append(0)
            else:
                re.append(sum(obj_dir)/len(obj_dir))        
        return np.array(re).reshape(len(re),1)

    def cal_feature_ave_y_speed(self,s_index,dir): 
        re= []
        seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]        
        for d in seque_data:
            obj_num =0 
            obj_dir = [obj.vY/100 for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)] #速度转换�????1km/h 或�?1m/s
            if len(obj_dir)==0:
                re.append(0)
            else:
                re.append(sum(obj_dir)/len(obj_dir))        
        return np.array(re).reshape(len(re),1)


    def cal_feature_ave_speed_rate(self,s_index,dir):    
        re= []
        rePreN= []
        seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]
        seque_data_preN = self.raw_data[s_index-self.frame_diff_N:s_index+self.feature_seque_len-self.frame_diff_N]
        for d in seque_data:
            obj_dir = [obj.speed/100 for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)]
            if len(obj_dir)==0:
                re.append(0)
            else:
                re.append(sum(obj_dir)/len(obj_dir))     
        for d in seque_data_preN:
            obj_dir_preN = [obj.speed/100 for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)]
            if len(obj_dir_preN)==0:
                rePreN.append(0)
            else:
                rePreN.append(sum(obj_dir_preN)/len(obj_dir_preN))   
        return (np.array(re)-np.array(rePreN)).reshape(len(re),1)

    def cal_feature_ave_x_speed_rate(self,s_index,dir):    
        re= []
        rePreN= []
        seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]
        seque_data_preN = self.raw_data[s_index-self.frame_diff_N:s_index+self.feature_seque_len-self.frame_diff_N]
        for d in seque_data:
            obj_dir = [obj.vX/100 for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)]
            if len(obj_dir)==0:
                re.append(0)
            else:
                re.append(sum(obj_dir)/len(obj_dir))     
        for d in seque_data_preN:
            obj_dir_preN = [obj.vX/100 for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)]
            if len(obj_dir_preN)==0:
                rePreN.append(0)
            else:
                rePreN.append(sum(obj_dir_preN)/len(obj_dir_preN))   
        return (np.array(re)-np.array(rePreN)).reshape(len(re),1)    

    def cal_feature_ave_y_speed_rate(self,s_index,dir):    
        re= []
        rePreN= []
        seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]
        seque_data_preN = self.raw_data[s_index-self.frame_diff_N:s_index+self.feature_seque_len-self.frame_diff_N]
        for d in seque_data:
            obj_dir = [obj.vY/100 for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)]
            if len(obj_dir)==0:
                re.append(0)
            else:
                re.append(sum(obj_dir)/len(obj_dir))     
        for d in seque_data_preN:
            obj_dir_preN = [obj.vY/100 for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)]
            if len(obj_dir_preN)==0:
                rePreN.append(0)
            else:
                rePreN.append(sum(obj_dir_preN)/len(obj_dir_preN))   
        return (np.array(re)-np.array(rePreN)).reshape(len(re),1)    




    def cal_feature_speed_std(self,s_index,dir):   #标准�????
        re= []
        seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]        
        for d in seque_data:
            obj_num =0 
            obj_dir = [obj.speed/100 for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)] #速度转换�????1km/h 或�?1m/s
            if len(obj_dir)==0:
                re.append(0)
            else:
                re.append(np.std(obj_dir))        
        return np.array(re).reshape(len(re),1)
    def cal_feature_x_speed_std(self,s_index,dir):   #标准�????
        re= []
        seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]        
        for d in seque_data:
            obj_num =0 
            obj_dir = [obj.vX/100 for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)] #速度转换�????1km/h 或�?1m/s
            if len(obj_dir)==0:
                re.append(0)
            else:
                re.append(np.std(obj_dir))        
        return np.array(re).reshape(len(re),1)
    def cal_feature_y_speed_std(self,s_index,dir):   #标准�????
        re= []
        seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]        
        for d in seque_data:
            obj_num =0 
            obj_dir = [obj.vY/100 for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)] #速度转换�????1km/h 或�?1m/s
            if len(obj_dir)==0:
                re.append(0)
            else:
                re.append(np.std(obj_dir))        
        return np.array(re).reshape(len(re),1)
    def cal_feature_obj_num(self,s_index,dir):        
        re= []
        
        seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]        
        for d in seque_data:
            obj_num =0 
            obj_dir = [obj for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)]
            re.append(len(obj_dir))        
        return np.array(re).reshape(len(re),1)
    
    def cal_feature_obj_num_rate(self,s_index,dir):        
        re= []
        rePreN= []
        seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]
        seque_data_preN = self.raw_data[s_index-self.frame_diff_N:s_index+self.feature_seque_len-self.frame_diff_N]
        for d in seque_data:
            obj_dir = [obj for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)]
            re.append(len(obj_dir))    
        for d in seque_data_preN:
            obj_dir_preN = [obj for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)]
            rePreN.append(len(obj_dir_preN))   
        return (np.array(re)-np.array(rePreN)).reshape(len(re),1)

    def cal_feature_obj_num_high_speed(self,s_index,dir):        
        re= []
        seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]        
        for d in seque_data:
            obj_num =0 
            obj_dir = [obj for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)]
            obj_dir_highspeed = [obj for obj in obj_dir if obj.speed>self.speed_th]
            re.append(len(obj_dir_highspeed))        
        return np.array(re).reshape(len(re),1)

    def cal_feature_obj_num_high_speed_rate(self,s_index,dir):        
        re= []
        rePreN= []
        seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]
        seque_data_preN = self.raw_data[s_index-self.frame_diff_N:s_index+self.feature_seque_len-self.frame_diff_N]
        for d in seque_data:
            obj_dir = [obj for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10) and obj.speed>self.speed_th]
            re.append(len(obj_dir))    
        for d in seque_data_preN:
            obj_dir_preN = [obj for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10) and obj.speed>self.speed_th]
            rePreN.append(len(obj_dir_preN))   
        return (np.array(re)-np.array(rePreN)).reshape(len(re),1)

    def cal_feature_obj_num_low_speed(self,s_index,dir):        
        re= []
        seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]        
        for d in seque_data:
            obj_num =0 
            obj_dir = [obj for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)]
            obj_dir_highspeed = [obj for obj in obj_dir if obj.speed<=self.speed_th]
            re.append(len(obj_dir_highspeed))        
        return np.array(re).reshape(len(re),1)

    def cal_feature_obj_num_low_speed_rate(self,s_index,dir):        
        re= []
        rePreN= []
        seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]
        seque_data_preN = self.raw_data[s_index-self.frame_diff_N:s_index+self.feature_seque_len-self.frame_diff_N]
        for d in seque_data:
            obj_dir = [obj for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10) and obj.speed<=self.speed_th]
            re.append(len(obj_dir))    
        for d in seque_data_preN:
            obj_dir_preN = [obj for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10) and obj.speed<=self.speed_th]
            rePreN.append(len(obj_dir_preN))   
        return (np.array(re)-np.array(rePreN)).reshape(len(re),1)
    def cal_feature_avg_nearest_X_dist(self,s_index,dir): 
        re= []
        seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]        
        for d in seque_data:
            obj_num =0 
            obj_dir = [obj.x/100 for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)] #距离转换为m
            obj_dir_closest = find_closest_numbers(obj_dir)  
            if len(obj_dir)==0:
                re.append(0)
            elif len(obj_dir)==1:
                re.append(100)
            else:
                re.append(sum(obj_dir_closest)/len(obj_dir_closest))        
        return np.array(re).reshape(len(re),1)
    
    def cal_feature_avg_nearest_Y_dist(self,s_index,dir): 
        re= []
        seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]        
        for d in seque_data:
            obj_num =0 
            obj_dir = [obj.y/100 for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)] #距离转换为m
            obj_dir_closest = find_closest_numbers(obj_dir)  
            if len(obj_dir)==0:
                re.append(0)
            elif len(obj_dir)==1:
                re.append(100)
            else:
                re.append(sum(obj_dir_closest)/len(obj_dir_closest))        
        return np.array(re).reshape(len(re),1)

    def cal_feature_avg_nearest_XY_dist(self,s_index,dir): 
        re= []
        seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]        
        for d in seque_data:
            obj_num =0 
            obj_dir = [(obj.y/100*obj.y/100*+obj.x/100*obj.x/100)**0.5 for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)] #距离转换为m
            obj_dir_closest = find_closest_numbers(obj_dir)  
            if len(obj_dir)==0:
                re.append(0)
            elif len(obj_dir)==1:
                re.append(100)
            else:
                re.append(sum(obj_dir_closest)/len(obj_dir_closest))        
        return np.array(re).reshape(len(re),1)
    def cal_feature_lane_speed_std(self,s_index,dir): 
        re= []
        seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]        
        for d in seque_data:
            lane_speed_dict={}
            for obj in d.obj_info:
                if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10):
                    if obj.obj_lanenum not in lane_speed_dict.keys():
                        lane_speed_dict[obj.obj_lanenum] = [obj.speed/100]
                    else:
                        lane_speed_dict[obj.obj_lanenum].append(obj.speed/100)
            each_lane_avgspeed=[]
            for lane_speed in lane_speed_dict.values():
                each_lane_avgspeed.append(sum(lane_speed)/len(lane_speed))
            if len(each_lane_avgspeed)==0:
                re.append(0)
            else:
                re.append(np.std(each_lane_avgspeed))  
        return np.array(re).reshape(len(re),1)
    def cal_feature_lane_xspeed_std(self,s_index,dir): 
        re= []
        seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]        
        for d in seque_data:
            lane_speed_dict={}
            for obj in d.obj_info:
                if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10):
                    if obj.obj_lanenum not in lane_speed_dict.keys():
                        lane_speed_dict[obj.obj_lanenum] = [obj.vX/100]
                    else:
                        lane_speed_dict[obj.obj_lanenum].append(obj.vX/100)
            each_lane_avgspeed=[]
            for lane_speed in lane_speed_dict.values():
                each_lane_avgspeed.append(sum(lane_speed)/len(lane_speed))
            if len(each_lane_avgspeed)==0:
                re.append(0)
            else:
                re.append(np.std(each_lane_avgspeed))  
        return np.array(re).reshape(len(re),1)

    def cal_feature_lane_yspeed_std(self,s_index,dir): 
        re= []
        seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]        
        for d in seque_data:
            lane_speed_dict={}
            for obj in d.obj_info:
                if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10):
                    if obj.obj_lanenum not in lane_speed_dict.keys():
                        lane_speed_dict[obj.obj_lanenum] = [obj.vY/100]
                    else:
                        lane_speed_dict[obj.obj_lanenum].append(obj.vY/100)
            each_lane_avgspeed=[]
            for lane_speed in lane_speed_dict.values():
                each_lane_avgspeed.append(sum(lane_speed)/len(lane_speed))
            if len(each_lane_avgspeed)==0:
                re.append(0)
            else:
                re.append(np.std(each_lane_avgspeed))  
        return np.array(re).reshape(len(re),1)
    def cal_feature_lane_objnum_std(self,s_index,dir): 
        re= []
        seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]        
        for d in seque_data:

            lane_num_dict={}
            for obj in d.obj_info:
                if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10):
                    if obj.obj_lanenum not in lane_num_dict.keys():
                        lane_num_dict[obj.obj_lanenum] = 1
                    else:
                        lane_num_dict[obj.obj_lanenum]+=1
            each_lane_avgspeed=[]
            for lane_obj_num in lane_num_dict.values():
                each_lane_avgspeed.append(lane_obj_num)
            if len(each_lane_avgspeed)==0:
                re.append(0)
            else:
                re.append(np.std(each_lane_avgspeed))  
        return np.array(re).reshape(len(re),1)

    def cal_feature(self,dirs):        
        index = self.index  
        indexend = self.indexend

        while index<len(self.raw_data):     
            if self.raw_data_time[index]>self.end_time:
                break      
            print(index,"/",indexend)
            if index <self.feature_seque_len+self.frame_diff_N:
                index = self.feature_seque_len+self.frame_diff_N
            s_index = index-self.feature_seque_len
            dir_f={}
            for name in self.feature_name:
                for dir in dirs:
                    f_d = self.feature_func_dict[name](s_index,dir)
                    if dir not in dir_f.keys():
                        dir_f[dir] = f_d.reshape(self.feature_seque_len,1)
                    else:
                        dir_f[dir] = np.concatenate((dir_f[dir] , f_d), axis=1)
                    # print(dir_f[dir].shape)
                    None    
            for dir in dir_f.keys():
                if dir not in self.feature_data_dict.keys():
                    self.feature_data_dict[dir] = dir_f[dir].reshape(1,self.feature_seque_len,self.feature_dim)
                    self.feature_time_dict[dir] = [self.raw_data_time[index].strftime('%Y%m%d_%H%M%S%f')[:-3]]
                else:
                    self.feature_data_dict[dir] = np.concatenate((self.feature_data_dict[dir], dir_f[dir].reshape(1,self.feature_seque_len,self.feature_dim)), axis=0)
                    self.feature_time_dict[dir].append(self.raw_data_time[index].strftime('%Y%m%d_%H%M%S%f')[:-3])
            index +=self.feature_seque_offset


                    




        # print([d.time for d in seque_data])
        # print(self.raw_data_time[index])
        # print(index)

        None


    def get_raw_data(self,path):
        pre_time = self.start_time - datetime.timedelta(hours=1)        
        current_time = pre_time
        raw_data=[]
        while current_time <= self.end_time:
            # print(current_time)
            pkl_file = path+current_time.strftime('%Y%m%d_%H')+'.pkl'
            self.raw_data_file_name.append(pkl_file)            
            current_time += datetime.timedelta(hours=1)
        pkl_file = path+self.end_time.strftime('%Y%m%d_%H')+'.pkl'
        if pkl_file not in self.raw_data_file_name:
            self.raw_data_file_name.append(pkl_file)        
        for pkl_file in self.raw_data_file_name:
            if not os.path.exists(pkl_file):
                continue
            with open(pkl_file, 'rb') as file:
                
                loaded_data = pickle.load(file)
                raw_data = raw_data+loaded_data   
                print(pkl_file) 
        return raw_data    

class Radar_Feature_CNN:
    def __init__(self,start_time:datetime.datetime,end_time:datetime.datetime,path) -> None:
        self.start_time=start_time
        self.end_time = end_time
        # self.feature_dim = len(feature_names)
        
        self.feature_func_dict={}
        self.feature_func_dict['obj_num']=self.cal_feature_obj_num
        # self.feature_func_dict['obj_num_rate']=self.cal_feature_obj_num_rate
        self.feature_func_dict['obj_num_highspeed']=self.cal_feature_obj_num_high_speed
        # self.feature_func_dict['obj_num_highspeed_rate']=self.cal_feature_obj_num_high_speed_rate
        self.feature_func_dict['obj_num_lowspeed']=self.cal_feature_obj_num_low_speed
        # self.feature_func_dict['obj_num_lowspeed_rate']=self.cal_feature_obj_num_low_speed_rate
        self.feature_func_dict['obj_avespeed']=self.cal_feature_ave_speed
        self.feature_func_dict['obj_aveaccX']=self.cal_feature_ave_accX
        self.feature_func_dict['obj_aveaccY']=self.cal_feature_ave_accY       

        self.feature_func_dict['obj_avexspeed']=self.cal_feature_ave_x_speed
        self.feature_func_dict['obj_aveyspeed']=self.cal_feature_ave_y_speed
        # self.feature_func_dict['obj_avespeed_rate']=self.cal_feature_ave_speed_rate
        # self.feature_func_dict['obj_avexspeed_rate']=self.cal_feature_ave_x_speed_rate
        # self.feature_func_dict['obj_aveyspeed_rate']=self.cal_feature_ave_y_speed_rate

        self.feature_func_dict['obj_speed_std']=self.cal_feature_speed_std
        self.feature_func_dict['obj_xspeed_std']=self.cal_feature_x_speed_std
        self.feature_func_dict['obj_yspeed_std']=self.cal_feature_y_speed_std        

        self.feature_func_dict['obj_avg_nearest_X_dist']=self.cal_feature_avg_nearest_X_dist
        self.feature_func_dict['obj_avg_nearest_Y_dist']=self.cal_feature_avg_nearest_Y_dist
        self.feature_func_dict['obj_avg_nearest_XY_dist']=self.cal_feature_avg_nearest_XY_dist
        self.feature_func_dict['lane_speed_std']=self.cal_feature_lane_speed_std
        self.feature_func_dict['lane_xspeed_std']=self.cal_feature_lane_xspeed_std
        self.feature_func_dict['lane_yspeed_std']=self.cal_feature_lane_yspeed_std

        self.feature_func_dict['lane_objnum_std']=self.cal_feature_lane_objnum_std
        # self.feature_data_dict['is_lane_in']=self.cal_feature_lane_is_in
        # self.feature_data_dict['lane_color']=self.cal_feature_lane_color

        
        self.feature_name = list(self.feature_func_dict.keys())[:]
        self.speed_th = 1000
        self.frame_diff_N = 1000

        


        self.feature_dim = len(self.feature_name)
        self.feature_seque_len = 36  # 24个特征组成一个数�??
        self.feature_seque_offset = 3 #  5 帧算一个特�??

        self.raw_data_file_name=[]
        self.raw_data:List[Radar_Dat] = self.get_raw_data(path)
        self.raw_data_time = [item.time for item in self.raw_data]
        self.feature_time_dict={}
        self.feature_data_dict={}
        self.index = bisect.bisect(self.raw_data_time, self.start_time)
        self.indexend = bisect.bisect(self.raw_data_time, self.end_time)
        # self.cal_feature([11,31,51,71])
        None
    
    def cal_feature_lane_color(self,s_index,dir):

        lane_color_set={}
        lane_color_set[1]=[1,2,3,4,5,22,23,24]
        lane_color_set[2]=[6,7,8,14,17,18,19,20,21]
        lane_light_color_set={}
        re= []
        seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]        
        for d in seque_data:
            for lane in d.lane_info:

                for k,v in lane_color_set.items():

                    if lane.no in v:
                        if k not in lane_light_color_set.keys():
                            lane_light_color_set[k]=[lane.color]
                        else:
                            lane_light_color_set[k].append(lane.color)

                

                if lane.no==dir:
                    re.append(lane.color)        

    def cal_feature_lane_is_in(self,s_index,dir):
        re= []
        seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]        
        for d in seque_data:
            re.append(dir%10)    
        return np.array(re).reshape(len(re),1)
    
    def cal_feature_ave_speed(self,s_index,dir): 
        re= []       
        # seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]    
        for i in range(self.feature_seque_len): 
            seque_data = self.raw_data[s_index+i*self.feature_seque_offset:s_index+self.feature_seque_offset*(i+1)]
            obj_speed =[] 
            for d in seque_data:                
                obj_dir = [obj.speed/100 for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)]                
                obj_speed+=obj_dir
            if len(obj_speed)==0:
                re.append(0)
            else:
                re.append(sum(obj_speed)/len(obj_speed))
        return np.array(re).reshape(len(re),1)   

    def cal_feature_ave_accX(self,s_index,dir): 
        re= []       
        # seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]    
        for i in range(self.feature_seque_len): 
            seque_data = self.raw_data[s_index+i*self.feature_seque_offset:s_index+self.feature_seque_offset*(i+1)]
            obj_speed =[] 
            for d in seque_data:                
                obj_dir = [obj.accX/100 for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)]                
                obj_speed+=obj_dir
            if len(obj_speed)==0:
                re.append(0)
            else:
                re.append(sum(obj_speed)/len(obj_speed))
        return np.array(re).reshape(len(re),1) 
    def cal_feature_ave_accY(self,s_index,dir): 
        re= []       
        # seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]    
        for i in range(self.feature_seque_len): 
            seque_data = self.raw_data[s_index+i*self.feature_seque_offset:s_index+self.feature_seque_offset*(i+1)]
            obj_speed =[] 
            for d in seque_data:                
                obj_dir = [obj.axxY/100 for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)]                
                obj_speed+=obj_dir
            if len(obj_speed)==0:
                re.append(0)
            else:
                re.append(sum(obj_speed)/len(obj_speed))
        return np.array(re).reshape(len(re),1) 
    def cal_feature_ave_x_speed(self,s_index,dir): 
        re= []       
        # seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]    
        for i in range(self.feature_seque_len): 
            seque_data = self.raw_data[s_index+i*self.feature_seque_offset:s_index+self.feature_seque_offset*(i+1)]
            obj_speed =[] 
            for d in seque_data:                
                obj_dir = [obj.vX/100 for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)]                
                obj_speed+=obj_dir
            if len(obj_speed)==0:
                re.append(0)                
            else:
                re.append(sum(obj_speed)/len(obj_speed))
        return np.array(re).reshape(len(re),1) 

    def cal_feature_ave_y_speed(self,s_index,dir): 
        re= []       
        # seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]    
        for i in range(self.feature_seque_len): 
            seque_data = self.raw_data[s_index+i*self.feature_seque_offset:s_index+self.feature_seque_offset*(i+1)]
            obj_speed =[] 
            for d in seque_data:                
                obj_dir = [obj.vY/100 for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)]                
                obj_speed+=obj_dir
            if len(obj_speed)==0:
                re.append(0)
            else:
                re.append(sum(obj_speed)/len(obj_speed))
                
        return np.array(re).reshape(len(re),1) 


    def cal_feature_ave_speed_rate(self,s_index,dir):    
        re= []
        rePreN= []
        seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]
        seque_data_preN = self.raw_data[s_index-self.frame_diff_N:s_index+self.feature_seque_len-self.frame_diff_N]
        for d in seque_data:
            obj_dir = [obj.speed/100 for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)]
            if len(obj_dir)==0:
                re.append(0)
            else:
                re.append(sum(obj_dir)/len(obj_dir))     
        for d in seque_data_preN:
            obj_dir_preN = [obj.speed/100 for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)]
            if len(obj_dir_preN)==0:
                rePreN.append(0)
            else:
                rePreN.append(sum(obj_dir_preN)/len(obj_dir_preN))   
        return (np.array(re)-np.array(rePreN)).reshape(len(re),1)

    def cal_feature_ave_x_speed_rate(self,s_index,dir):    
        re= []
        rePreN= []
        seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]
        seque_data_preN = self.raw_data[s_index-self.frame_diff_N:s_index+self.feature_seque_len-self.frame_diff_N]
        for d in seque_data:
            obj_dir = [obj.vX/100 for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)]
            if len(obj_dir)==0:
                re.append(0)
            else:
                re.append(sum(obj_dir)/len(obj_dir))     
        for d in seque_data_preN:
            obj_dir_preN = [obj.vX/100 for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)]
            if len(obj_dir_preN)==0:
                rePreN.append(0)
            else:
                rePreN.append(sum(obj_dir_preN)/len(obj_dir_preN))   
        return (np.array(re)-np.array(rePreN)).reshape(len(re),1)    

    def cal_feature_ave_y_speed_rate(self,s_index,dir):    
        re= []
        rePreN= []
        seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]
        seque_data_preN = self.raw_data[s_index-self.frame_diff_N:s_index+self.feature_seque_len-self.frame_diff_N]
        for d in seque_data:
            obj_dir = [obj.vY/100 for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)]
            if len(obj_dir)==0:
                re.append(0)
            else:
                re.append(sum(obj_dir)/len(obj_dir))     
        for d in seque_data_preN:
            obj_dir_preN = [obj.vY/100 for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)]
            if len(obj_dir_preN)==0:
                rePreN.append(0)
            else:
                rePreN.append(sum(obj_dir_preN)/len(obj_dir_preN))   
        return (np.array(re)-np.array(rePreN)).reshape(len(re),1)    




    def cal_feature_speed_std(self,s_index,dir):   #标准
        re= []       
        # seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]    
        for i in range(self.feature_seque_len): 
            seque_data = self.raw_data[s_index+i*self.feature_seque_offset:s_index+self.feature_seque_offset*(i+1)]
            obj_speed =[] 
            for d in seque_data:                
                obj_dir = [obj.speed/100 for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)]                
                obj_speed+=obj_dir
            if len(obj_speed)==0:
                re.append(0)
            else:
                re.append(np.std(obj_speed))
        return np.array(re).reshape(len(re),1) 

    def cal_feature_x_speed_std(self,s_index,dir):   #标准�????
        re= []       
        # seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]    
        for i in range(self.feature_seque_len): 
            seque_data = self.raw_data[s_index+i*self.feature_seque_offset:s_index+self.feature_seque_offset*(i+1)]
            obj_speed =[] 
            for d in seque_data:                
                obj_dir = [obj.vX/100 for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)]                
                obj_speed+=obj_dir
            if len(obj_speed)==0:
                re.append(0)
            else:
                re.append(np.std(obj_speed))
        return np.array(re).reshape(len(re),1) 
    def cal_feature_y_speed_std(self,s_index,dir):   #标准�????
        re= []       
        # seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]    
        for i in range(self.feature_seque_len): 
            seque_data = self.raw_data[s_index+i*self.feature_seque_offset:s_index+self.feature_seque_offset*(i+1)]
            obj_speed =[] 
            for d in seque_data:                
                obj_dir = [obj.vY/100 for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)]                
                obj_speed+=obj_dir
            if len(obj_speed)==0:
                re.append(0)
            else:
                re.append(np.std(obj_speed))
        return np.array(re).reshape(len(re),1) 
    def cal_feature_obj_num(self,s_index,dir):        
        re= []       
        # seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]    
        for i in range(self.feature_seque_len): 
            seque_data = self.raw_data[s_index+i*self.feature_seque_offset:s_index+self.feature_seque_offset*(i+1)]
            obj_num =0 
            for d in seque_data:                
                obj_dir = [obj for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)]
                obj_num += len(obj_dir)
            re.append(obj_num)        
        return np.array(re).reshape(len(re),1)
    
    def cal_feature_obj_num_rate(self,s_index,dir):        
        re= []
        rePreN= []
        seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]
        seque_data_preN = self.raw_data[s_index-self.frame_diff_N:s_index+self.feature_seque_len-self.frame_diff_N]
        for d in seque_data:
            obj_dir = [obj for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)]
            re.append(len(obj_dir))    
        for d in seque_data_preN:
            obj_dir_preN = [obj for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)]
            rePreN.append(len(obj_dir_preN))   
        return (np.array(re)-np.array(rePreN)).reshape(len(re),1)

    def cal_feature_obj_num_high_speed(self,s_index,dir):
        re= []       
        # seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]    
        for i in range(self.feature_seque_len): 
            seque_data = self.raw_data[s_index+i*self.feature_seque_offset:s_index+self.feature_seque_offset*(i+1)]
            obj_num =0 
            for d in seque_data:                
                obj_dir = [obj for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)]
                obj_dir_highspeed = [obj for obj in obj_dir if obj.speed>self.speed_th]
                obj_num += len(obj_dir_highspeed)
            re.append(obj_num)        
        return np.array(re).reshape(len(re),1)        

    def cal_feature_obj_num_high_speed_rate(self,s_index,dir):        
        re= []
        rePreN= []
        seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]
        seque_data_preN = self.raw_data[s_index-self.frame_diff_N:s_index+self.feature_seque_len-self.frame_diff_N]
        for d in seque_data:
            obj_dir = [obj for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10) and obj.speed>self.speed_th]
            re.append(len(obj_dir))    
        for d in seque_data_preN:
            obj_dir_preN = [obj for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10) and obj.speed>self.speed_th]
            rePreN.append(len(obj_dir_preN))   
        return (np.array(re)-np.array(rePreN)).reshape(len(re),1)

    def cal_feature_obj_num_low_speed(self,s_index,dir):   
        re= []       
        # seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]    
        for i in range(self.feature_seque_len): 
            seque_data = self.raw_data[s_index+i*self.feature_seque_offset:s_index+self.feature_seque_offset*(i+1)]
            obj_num =0 
            for d in seque_data:                
                obj_dir = [obj for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)]
                obj_dir_highspeed = [obj for obj in obj_dir if obj.speed<=self.speed_th]
                obj_num += len(obj_dir_highspeed)
            re.append(obj_num)        
        return np.array(re).reshape(len(re),1)       

    def cal_feature_obj_num_low_speed_rate(self,s_index,dir):        
        re= []
        rePreN= []
        seque_data = self.raw_data[s_index:s_index+self.feature_seque_len]
        seque_data_preN = self.raw_data[s_index-self.frame_diff_N:s_index+self.feature_seque_len-self.frame_diff_N]
        for d in seque_data:
            obj_dir = [obj for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10) and obj.speed<=self.speed_th]
            re.append(len(obj_dir))    
        for d in seque_data_preN:
            obj_dir_preN = [obj for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10) and obj.speed<=self.speed_th]
            rePreN.append(len(obj_dir_preN))   
        return (np.array(re)-np.array(rePreN)).reshape(len(re),1)
    def cal_feature_avg_nearest_X_dist(self,s_index,dir):
        re= []         
        for i in range(self.feature_seque_len): 
            seque_data = self.raw_data[s_index+i*self.feature_seque_offset:s_index+self.feature_seque_offset*(i+1)]
            obj_list = [] 
            for d in seque_data:                
                obj_dir = [obj.x/100  for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)]
                obj_list+=obj_dir
            obj_dir_closest = find_closest_numbers(obj_list) 
            if len(obj_list)==0:
                re.append(0)
            elif len(obj_list)==1:
                re.append(100)
            else:
                re.append(sum(obj_dir_closest)/len(obj_dir_closest))                    
        return np.array(re).reshape(len(re),1)    
    
    def cal_feature_avg_nearest_Y_dist(self,s_index,dir): 
        re= []         
        for i in range(self.feature_seque_len): 
            seque_data = self.raw_data[s_index+i*self.feature_seque_offset:s_index+self.feature_seque_offset*(i+1)]
            obj_list = [] 
            for d in seque_data:                
                obj_dir = [obj.y/100  for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)]
                obj_list+=obj_dir
            obj_dir_closest = find_closest_numbers(obj_list) 
            if len(obj_list)==0:
                re.append(0)
            elif len(obj_list)==1:
                re.append(100)
            else:
                re.append(sum(obj_dir_closest)/len(obj_dir_closest))                    
        return np.array(re).reshape(len(re),1)  

    def cal_feature_avg_nearest_XY_dist(self,s_index,dir): 
        re= []         
        for i in range(self.feature_seque_len): 
            seque_data = self.raw_data[s_index+i*self.feature_seque_offset:s_index+self.feature_seque_offset*(i+1)]
            obj_list = [] 
            for d in seque_data:                
                obj_dir = [(obj.y/100*obj.y/100*+obj.x/100*obj.x/100)**0.5 for obj in d.obj_info if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10)]
                obj_list+=obj_dir
            obj_dir_closest = find_closest_numbers(obj_list) 
            if len(obj_list)==0:
                re.append(0)
            elif len(obj_list)==1:
                re.append(100)
            else:
                re.append(sum(obj_dir_closest)/len(obj_dir_closest))                    
        return np.array(re).reshape(len(re),1)  

    def cal_feature_lane_speed_std(self,s_index,dir):
        re= []                
        for i in range(self.feature_seque_len): 
            lane_speed_dict={} 
            seque_data = self.raw_data[s_index+i*self.feature_seque_offset:s_index+self.feature_seque_offset*(i+1)]
            for d in seque_data:
                for obj in d.obj_info:
                    if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10):
                        if obj.obj_lanenum not in lane_speed_dict.keys():
                            lane_speed_dict[obj.obj_lanenum] = [obj.speed/100]
                        else:
                            lane_speed_dict[obj.obj_lanenum].append(obj.speed/100)
            
            each_lane_avgspeed=[]
            for lane_speed in lane_speed_dict.values():
                each_lane_avgspeed.append(sum(lane_speed)/len(lane_speed))                
            if len(each_lane_avgspeed)==0:
                re.append(0)
            else:
                re.append(np.std(each_lane_avgspeed)) 
        return np.array(re).reshape(len(re),1)        
    
    def cal_feature_lane_xspeed_std(self,s_index,dir): 
        re= []                
        for i in range(self.feature_seque_len): 
            lane_speed_dict={} 
            seque_data = self.raw_data[s_index+i*self.feature_seque_offset:s_index+self.feature_seque_offset*(i+1)]
            for d in seque_data:
                for obj in d.obj_info:
                    if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10):
                        if obj.obj_lanenum not in lane_speed_dict.keys():
                            lane_speed_dict[obj.obj_lanenum] = [obj.vX/100]
                        else:
                            lane_speed_dict[obj.obj_lanenum].append(obj.vX/100)
            
            each_lane_avgspeed=[]
            for lane_speed in lane_speed_dict.values():
                each_lane_avgspeed.append(sum(lane_speed)/len(lane_speed))                
            if len(each_lane_avgspeed)==0:
                re.append(0)
            else:
                re.append(np.std(each_lane_avgspeed)) 
        return np.array(re).reshape(len(re),1)     
        
    def cal_feature_lane_yspeed_std(self,s_index,dir): 
        re= []                
        for i in range(self.feature_seque_len): 
            lane_speed_dict={} 
            seque_data = self.raw_data[s_index+i*self.feature_seque_offset:s_index+self.feature_seque_offset*(i+1)]
            for d in seque_data:
                for obj in d.obj_info:
                    if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10):
                        if obj.obj_lanenum not in lane_speed_dict.keys():
                            lane_speed_dict[obj.obj_lanenum] = [obj.vY/100]
                        else:
                            lane_speed_dict[obj.obj_lanenum].append(obj.vY/100)
            
            each_lane_avgspeed=[]
            for lane_speed in lane_speed_dict.values():
                each_lane_avgspeed.append(sum(lane_speed)/len(lane_speed))                
            if len(each_lane_avgspeed)==0:
                re.append(0)
            else:
                re.append(np.std(each_lane_avgspeed)) 
        return np.array(re).reshape(len(re),1)     
    def cal_feature_lane_objnum_std(self,s_index,dir): 
        re= []                
        for i in range(self.feature_seque_len): 
            lane_speed_dict={} 
            seque_data = self.raw_data[s_index+i*self.feature_seque_offset:s_index+self.feature_seque_offset*(i+1)]
            for d in seque_data:
                for obj in d.obj_info:
                    if (obj.radar_dir==dir//10 and obj.is_in_lane==dir%10):
                        if obj.obj_lanenum not in lane_speed_dict.keys():
                            lane_speed_dict[obj.obj_lanenum] = 1
                        else:
                            lane_speed_dict[obj.obj_lanenum] +=1
            
            each_lane_avgspeed=[]
            for lane_speed in lane_speed_dict.values():
                each_lane_avgspeed.append(lane_speed)                
            if len(each_lane_avgspeed)==0:
                re.append(0)
            else:
                re.append(np.std(each_lane_avgspeed)) 
        return np.array(re).reshape(len(re),1)    

    def cal_feature(self,dirs):        
        index = self.index  
        indexend = self.indexend

        while index<len(self.raw_data):     
            if self.raw_data_time[index]>self.end_time:
                break      
            print(index,"/",indexend)
            if index <self.feature_seque_len+self.frame_diff_N:
                index = self.feature_seque_len+self.frame_diff_N
            s_index = index-self.feature_seque_len
            dir_f={}
            for name in self.feature_name:
                for dir in dirs:
                    f_d = self.feature_func_dict[name](s_index,dir)
                    if dir not in dir_f.keys():
                        dir_f[dir] = f_d.reshape(self.feature_seque_len,1)
                    else:
                        dir_f[dir] = np.concatenate((dir_f[dir] , f_d), axis=1)
                    # print(dir_f[dir].shape)
                    None    
            for dir in dir_f.keys():
                if dir not in self.feature_data_dict.keys():
                    self.feature_data_dict[dir] = dir_f[dir].reshape(1,self.feature_seque_len,self.feature_dim)
                    self.feature_time_dict[dir] = [self.raw_data_time[index].strftime('%Y%m%d_%H%M%S%f')[:-3]]
                else:
                    self.feature_data_dict[dir] = np.concatenate((self.feature_data_dict[dir], dir_f[dir].reshape(1,self.feature_seque_len,self.feature_dim)), axis=0)
                    self.feature_time_dict[dir].append(self.raw_data_time[index].strftime('%Y%m%d_%H%M%S%f')[:-3])
            index +=self.feature_seque_offset


                    




        # print([d.time for d in seque_data])
        # print(self.raw_data_time[index])
        # print(index)

        None


    def get_raw_data(self,path):
        pre_time = self.start_time - datetime.timedelta(hours=1)        
        current_time = pre_time
        raw_data=[]
        while current_time <= self.end_time:
            print(current_time)
            pkl_file = path+current_time.strftime('%Y%m%d_%H')+'.pkl'
            self.raw_data_file_name.append(pkl_file)            
            current_time += datetime.timedelta(hours=1)
        pkl_file = path+self.end_time.strftime('%Y%m%d_%H')+'.pkl'
        if pkl_file not in self.raw_data_file_name:
            self.raw_data_file_name.append(pkl_file)        
        for pkl_file in self.raw_data_file_name:
            if not os.path.exists(pkl_file):
                continue
            with open(pkl_file, 'rb') as file:
                loaded_data = pickle.load(file)
                raw_data = raw_data+loaded_data    
        return raw_data    

class RadarFeatureReduction:
    def __init__(self,start_time:datetime.datetime,end_time:datetime.datetime,path) -> None:
        self.start_time=start_time
        self.end_time = end_time   
        self.raw_data_file_name=[]     
        self.raw_data:List[Radar_Dat] = self.get_raw_data(path)
        self.raw_data_time = [item.time for item in self.raw_data]
        self.index = bisect.bisect(self.raw_data_time, self.start_time)
        self.indexend = bisect.bisect(self.raw_data_time, self.end_time)
        
        None   
    def get_raw_data(self,path):
        pre_time = self.start_time - datetime.timedelta(hours=1)        
        current_time = pre_time
        raw_data=[]
        while current_time <= self.end_time:
            # print(current_time)
            pkl_file = path+current_time.strftime('%Y%m%d_%H')+'.pkl'
            self.raw_data_file_name.append(pkl_file)            
            current_time += datetime.timedelta(hours=1)
        pkl_file = path+self.end_time.strftime('%Y%m%d_%H')+'.pkl'
        if pkl_file not in self.raw_data_file_name:
            self.raw_data_file_name.append(pkl_file)        
        for pkl_file in self.raw_data_file_name:
            if not os.path.exists(pkl_file):
                continue
            with open(pkl_file, 'rb') as file:                
                loaded_data = pickle.load(file)
                raw_data = raw_data+loaded_data   
                print(pkl_file) 
        return raw_data    

    def get_lanecode_list(self,raw_data,dir,length=10000,th =30):

        dict_lanecode = {}
        if length>len(raw_data):
            length = len(raw_data)
            print("ERROR")
            print("Len of raw_data is "+str(len(raw_data))+" but length is "+str(length))
        max_lane_num = 0
        for frame in raw_data:
            for obj in frame.obj_info:
                if (obj.obj_lanenum<200 and obj.radar_dir == dir//10 and obj.is_in_lane==dir%10):
                    # if obj.obj_lanenum <6:
                    #     print("ERROR")
                    if obj.obj_lanenum not in dict_lanecode:
                        dict_lanecode[obj.obj_lanenum]=1
                    else:
                        dict_lanecode[obj.obj_lanenum]+=1
                    if dict_lanecode[obj.obj_lanenum]>max_lane_num:
                        max_lane_num = dict_lanecode[obj.obj_lanenum]
            if max_lane_num>10000:
                break
        if max_lane_num<10000:
            for i in range(100):
                print("ERROR")
                print("Max lane num is "+str(max_lane_num))
        print(dict_lanecode)
        re_list=[]
        for key in dict_lanecode.keys():
            if dict_lanecode[key]>max_lane_num/100:
                re_list.append(key)
                # del dict_lanecode[key]
        print(re_list)
        return re_list


class Radar_Dat:
    def __init__(self) -> None:
        self.time:datetime.datetime = None
        self.year=0
        self.month=0
        self.day =0
        self.hour = 0
        self.min=0
        self.sec = 0
        self.msec = 0        
        self.head_str=""
        self.lane_info=[Lane_Info]
        self.obj_info:List[Obj_Info]=[]
        self.obj_x_coords=[]
        self.obj_y_coords=[]

    def feature_cal(self):

        None

    def decode(self,frame):
        self.head_str = frame[0:12].decode('utf-8')
        self.year = frame[0+12]+1900
        self.month = frame[1+12]
        self.day = frame[2+12]
        self.hour = frame[3+12]
        self.min = frame[4+12]
        self.sec = frame[5+12]
        self.msec = int.from_bytes(frame[6+12:12+8], byteorder='little')  # 小端字节�????
        self.time = datetime.datetime(year=self.year,month=self.month,day=self.day,hour=self.hour,minute=self.min,second=self.sec,microsecond=self.msec*1000)
        self.lane_num=frame[20]
        self.lane_len=frame[21]
        self.lane_data_size = self.lane_num*self.lane_len
        for i in range(self.lane_num):
            lane_i = Lane_Info(frame=frame[22+self.lane_len*i:22+self.lane_len*(i+1)])
            self.lane_info.append(lane_i)
            None
        self.obj_num=int.from_bytes(frame[22+self.lane_data_size:24+self.lane_data_size],byteorder="little")
        self.obj_len =int.from_bytes(frame[24+self.lane_data_size:26+self.lane_data_size],byteorder="little")
        diff = 26+self.lane_data_size
        for i in range(self.obj_num):
            if len(frame[diff+self.obj_len*i:diff+self.obj_len*(i+1)])!=73:
                print("NNNNNNNNN",len(frame[diff+self.obj_len*i:diff+self.obj_len*(i+1)]))
                continue
            obj_i = Obj_Info(frame=frame[diff+self.obj_len*i:diff+self.obj_len*(i+1)])
            self.obj_info.append(obj_i)
        
        # self.obj_x_coords_dict={}

        self.obj_x_coords=[obj.x for obj in self.obj_info]
        self.obj_y_coords=[obj.y for obj in self.obj_info]
    def plot_obj(self,dir_list):

        for dir in dir_list:
            obj_info = [obj for obj in self.obj_info if obj.radar_dir == dir]
            obj_x_coords=[obj.x for obj in obj_info]
            obj_y_coords=[obj.y for obj in obj_info]
            plt.scatter(obj_x_coords,obj_y_coords)
            # 为每个点标注其坐�????
            for point in obj_info:
                plt.annotate(f"({point.x}, {point.y})", (point.x, point.y))

        # 显示图形
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(str(dir_list))
        plt.grid(True)
        plt.show(block=False)

    def plot_obj_d(self, dir_list,fig, ax,is_clear=0):

        # 清除子图
        if is_clear==1:
            ax.clear()
        # ax.set_xlim([0, 30000])
        # ax.set_ylim([-3000, 3000])
        for dir in dir_list:
            obj_info = [obj for obj in self.obj_info if ( obj.radar_dir == dir//10 and obj.is_in_lane==dir%10 and obj.obj_lanenum!=255)]
            
            # obj_info = [obj for obj in self.obj_info if (obj.radar_dir == dir)]
            obj_x_coords = [obj.x for obj in obj_info]
            obj_y_coords = [obj.y for obj in obj_info]
            ax.scatter(obj_x_coords, obj_y_coords)
            # 为每个点标注其坐
            # for point in obj_info:
            #     ax.annotate(f"{point.x},{point.y},{point.obj_lanenum}", (point.x, point.y))
        # datetime.datetime.strptime("")
        ax.set_title(str(self.time))
        
        # 显示更新后的
        plt.draw()
        plt.pause(0.001)


def find_files_withend(directory,end):
    dat_files = []

    # 遍历目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(end):
                dat_files.append(os.path.join(root, file))
        break

    return dat_files

def find_dat_files(directory):
    dat_files = []

    # 遍历目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.dat'):
                dat_files.append(os.path.join(root, file))

    return dat_files
def find_radar_data_files(directory):
    dat_files = []

    # 遍历目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # if file.endswith('.dat'):
            if file.startswith('radar_data'):
                dat_files.append(os.path.join(root, file))

    return dat_files

def read_frames_from_file(filename):
    frames = []
    cnt=0
    with open(filename, 'rb') as file:
        buffer = b""
        while True:
            byte = file.read(1)
            
            if not byte:
                # 文件结束时，如果缓冲区中有数据，将其作为最后一帧存�????
                if buffer:
                    frames.append(buffer)
                break

            buffer += byte
            if buffer[-12:] == b'[radar_data]':
                if len(buffer) > 12:  # 如果不是第一个帧
                    cnt +=1
                    # print(cnt)
                    # if cnt>10:
                    #     break
                    frames.append(buffer[:-12])  # 保存当前缓冲区中的数据，不包括新找到的标识符
                    buffer = buffer[-12:]  # 保留新找到的标识符作为下一个帧的开�????
    
    return frames

def read_dat_file(filename):
    with open(filename, 'rb') as file:
        # 假设每次都读�????4个字节，即一个int值（仅为示例�????
        while True:
            data = file.read(12)  # 读取4个字�????
            if not data:
                break
            # 假设是little-endian的整�????
            str_radar_code  = data.decode()
            value = int.from_bytes(data, byteorder='little')
            # value = str.from
            print(value)

def read_dat_to_pkl(file_path,pkl_path='./data/pkl/'):

    last_dir = file_path.split('/')[-1]

    dat_file_list = find_radar_data_files(file_path)
    dat_file_list = sorted(dat_file_list)
    pkl_dict={}
    last_date=None
    for dat_file in dat_file_list:
        frame_list = read_frames_from_file(dat_file)
        print(dat_file)
        for frame in frame_list:
            if len(frame)==5840:
                print(len(frame))
            radar_dat = Radar_Dat()
            try:
                radar_dat.decode(frame)
            except:
                print("decode error")
                continue
            time_str_toh = radar_dat.time.strftime('%Y%m%d_%H')
            if time_str_toh not in  pkl_dict.keys():
                if len(pkl_dict.keys())!=0:
                    pkl_save_path = pkl_path+last_dir+'/'
                    os.makedirs(pkl_save_path,exist_ok=True)
                    with open(pkl_save_path+last_date+'.pkl', 'wb') as file:
                        pickle.dump(pkl_dict[last_date], file)
                        print(pkl_save_path+last_date+'.pkl')
                        del pkl_dict[last_date] #删除，清理内

                pkl_dict[time_str_toh]=[]
                pkl_dict[time_str_toh].append(radar_dat)                
            else:
                pkl_dict[time_str_toh].append(radar_dat)
            last_date = time_str_toh
    if len(pkl_dict.keys())!=0:
        pkl_save_path = pkl_path+last_dir+'/'
        os.makedirs(pkl_save_path,exist_ok=True)
        with open(pkl_save_path+last_date+'.pkl', 'wb') as file:
            pickle.dump(pkl_dict[last_date], file)
            print(pkl_save_path+last_date+'.pkl')
            del pkl_dict[last_date] #删除，清理内

# if __name__ == "__main__":

#     # read_dat_to_pkl("./data/real/96/")

#     radar_dat_list : List[Radar_Dat]= []
#     with open('./data/real/161/pkl/20231024_19.pkl', 'rb') as file:
#         radar_dat_list = pickle.load(file)
#     for i in  range(len(radar_dat_list)):
#         radar_dat = radar_dat_list[i]
#         print(i,radar_dat.time)
#     None


def tsne_display(data):
    data_reshaped = data.reshape(data.shape[0], -1)  # 将数据重塑为 (N, 4200)
    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, random_state=42)
    data_tsne = tsne.fit_transform(data_reshaped)

    # 可视�????
    plt.figure(figsize=(10, 6))
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], alpha=0.5)
    plt.title("t-SNE visualization of the data")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.show()
def pcatsne_display1(data1, data2):
    # 合并数据
    merged_data = np.concatenate((data1, data2), axis=0)

    # 1. 重塑数据�???? (N, 4200)
    data_reshaped = merged_data.reshape(merged_data.shape[0], -1)

    # 2. 使用PCA进行初步降维
    pca = PCA(n_components=50)
    data_pca = pca.fit_transform(data_reshaped)

    # 3. 使用t-SNE进一步降�????
    tsne = TSNE(n_components=2, random_state=42)
    data_tsne = tsne.fit_transform(data_pca)

    # 4. 为两个数据集分配颜色
    colors = ['red'] * len(data1) + ['blue'] * len(data2)

    # 5. 可视�????
    plt.figure(figsize=(10, 6))
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=colors, alpha=0.5)
    plt.title("t-SNE visualization of the data")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.show()
def pcatsne_display(data):
    data_reshaped = data.reshape(data.shape[0], -1)  # 将数据重塑为 (N, 4200)
    # 使用t-SNE进行降维
# 2. 使用PCA进行初步降维
    pca = PCA(n_components=50)
    data_pca = pca.fit_transform(data_reshaped)

    # 3. 使用t-SNE进一步降�????
    tsne = TSNE(n_components=2, random_state=42)
    data_tsne = tsne.fit_transform(data_pca)

    # 4. 可视�????
    plt.figure(figsize=(10, 6))
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], alpha=0.5)
    plt.title("t-SNE visualization of the data")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.show()
def MI_cal(data):
    MI_values = []

    for t in range(data.shape[1]):
        X = data[:, t, 0].reshape(-1, 1)  # �????1个特�????
        Y = data[:, t, 1]                 # �????2个特�????
        MI = mutual_info_regression(X, Y)
        MI_values.append(MI[0])
    print(MI_values)


def pcatsne_display2(data_list):
    # 1. 合并数据
    merged_data = np.concatenate(data_list, axis=0)

    # 2. 重塑数据�???? (N, 4200)
    data_reshaped = merged_data.reshape(merged_data.shape[0], -1)

    # 3. 使用PCA进行初步降维
    pca = PCA(n_components=50)
    data_pca = pca.fit_transform(data_reshaped)

    # 4. 使用t-SNE进一步降
    tsne = TSNE(n_components=2, random_state=42)
    data_tsne = tsne.fit_transform(data_pca)

    # 5. 生成颜色列表
    colors = plt.cm.rainbow(np.linspace(0, 1, len(data_list)))
    
    # 6. 可视�????
    plt.figure(figsize=(10, 6))
    start_idx = 0
    for i, data in enumerate(data_list):
        plt.scatter(data_tsne[start_idx:start_idx+len(data), 0],
                    data_tsne[start_idx:start_idx+len(data), 1], 
                    color=colors[i], 
                    alpha=0.5, 
                    label=f'Dataset {i+1}')
        start_idx += len(data)

    plt.title("t-SNE visualization of the data")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    image_filename = f"tsne_{current_time}.png"
    plt.savefig(image_filename)
    plt.show()









# def compute_avg_MI(data):
#     N, T, F = data.shape  # 获取数据的形状，其中T应该�????300，F应该�????14
#     MI_matrices = []

#     for t in range(T):
#         MI_matrix_t = np.zeros((F, F))
#         for i in range(F):
#             for j in range(F):
#                 MI_matrix_t[i, j] = mutual_info_score(data[:, t, i], data[:, t, j])
#         MI_matrices.append(MI_matrix_t)

#     avg_MI_matrix = np.mean(MI_matrices, axis=0)
#     return avg_MI_matrix

def mutual_information(x, y, bins=30):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

def compute_avg_MI(data):
    N, T, F = data.shape  # 获取数据的形�????
    MI_matrices = []

    for t in range(T):
        MI_matrix_t = np.zeros((F, F))
        for i in range(F):
            for j in range(F):
                MI_matrix_t[i, j] = mutual_information(data[:, t, i], data[:, t, j])
        MI_matrices.append(MI_matrix_t)

    avg_MI_matrix = np.mean(MI_matrices, axis=0)
    return avg_MI_matrix

def display_the_origin_radar_data(start_time,end_time,path):


    radar_feature = Radar_Feature_CNN(start_time=start_time,end_time=end_time,path=path)
    # frame_list = read_frames_from_file("./data/real/161/20231023_190148.dat")
    

    # radar_dat_list = []
    fig, ax = plt.subplots()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show(block=False)


    for frame in radar_feature.raw_data[radar_feature.index:radar_feature.indexend]:
        # radar_dat = Radar_Dat()
        # radar_dat.decode(frame)
        # print(radar_dat.time)
        frame.plot_obj_d([1],fig, ax)
        # radar_dat.plot_obj(3)
        # radar_dat.plot_obj(5)
        # radar_dat.plot_obj(7)

        # radar_dat_list.append(radar_dat)

        None
    
    
    # with open('radar_data.pkl', 'wb') as file:
    #     pickle.dump(radar_dat_list, file)

    # with open('radar_data.pkl', 'rb') as file:
    #     loaded_data = pickle.load(file)
    # None

def display_the_origin_radar_data1(start_time,end_time,path,dir):


    radar_feature = Radar_Feature(start_time=start_time,end_time=end_time,path=path)
    fig, ax = plt.subplots()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show(block=False)
    for frame in radar_feature.raw_data[radar_feature.index:radar_feature.indexend]:
        frame.plot_obj_d([dir],fig, ax,is_clear=0)
        None
def display_the_origin_radar_data_once(start_time,end_time,path,dir):


    radar_feature = Radar_Feature(start_time=start_time,end_time=end_time,path=path)
    fig, ax = plt.subplots()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show(block=False)
    obj_x=[]
    obj_y=[]
    for frame in radar_feature.raw_data[radar_feature.index:radar_feature.indexend]:

        obj_info = [obj for obj in frame.obj_info if ( obj.radar_dir == dir//10 and obj.is_in_lane==dir%10 and obj.obj_lanenum!=255)]
        
        # obj_info = [obj for obj in self.obj_info if (obj.radar_dir == dir)]
        obj_x_coords = [obj.x for obj in obj_info]
        obj_y_coords = [obj.y for obj in obj_info]
        obj_x+=obj_x_coords
        obj_y+=obj_y_coords
        
    ax.scatter(obj_x, obj_y,s=1)
    plt.draw()
    plt.pause(500)
def display_the_history_trace(start_time,end_time,path,dir,label):
    radar_feature = Radar_Feature(start_time=start_time,end_time=end_time,path=path)
    fig, ax = plt.subplots()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    

    ax.clear()
    ax.set_xlim([-5000, 50000])
    ax.set_ylim([-5000, 5000])

    # dir=
    for frame in radar_feature.raw_data[radar_feature.index:radar_feature.indexend]:
        obj_info = [obj for obj in frame.obj_info if (obj.radar_dir == dir and obj.obj_lanenum!=255)]
        obj_x_coords = [obj.x for obj in obj_info]
        obj_y_coords = [obj.y for obj in obj_info]
        ax.scatter(obj_x_coords, obj_y_coords,s=1)
    s_time = start_time.strftime('%Y%m%d_%H%M%S')
    e_time = end_time.strftime('%Y%m%d_%H%M%S')
    ax.set_title(s_time+" : "+e_time)
    
    # 显示更新后的

    plt.draw()
    os.makedirs("./fig/",exist_ok=True)
    plt.savefig(f"./fig/"+label+"_{}_{}_{}.png".format(s_time,e_time,dir),dpi=300)
    #plt.show(block=True)
    # plt.pause(0.001)

 
def process_radar_data(start_time, end_time, dir, is_acc):
    start_time = datetime.datetime.strptime(start_time, "%Y%m%d_%H%M%S")
    end_time = datetime.datetime.strptime(end_time, "%Y%m%d_%H%M%S")
    
    radar_feature = Radar_Feature_CNN(start_time=start_time, end_time=end_time, path='./data/mmAcc/5008/pkl/')
    radar_feature.cal_feature([dir])
    
    save_name = f"./npy/{start_time.strftime('%Y%m%d_%H%M%S')}_{end_time.strftime('%Y%m%d_%H%M%S')}dir{int(dir//10)}_in{dir%10}_data_"
    
    if is_acc == 1:
        np.save(save_name + "acc.npy", radar_feature.feature_data_dict[dir])
    elif is_acc == 0:
        np.save(save_name + "normal.npy", radar_feature.feature_data_dict[dir])
    elif is_acc == -1:
        np.save(save_name + "inference.npy", radar_feature.feature_data_dict[dir])
    
    np.save(save_name + "time.npy", radar_feature.feature_time_dict[dir])

def main_get_inference_tst_dataset_mp():
    start_time_list = ["20231107_150000","20231107_150000", "20231107_150000", "20231107_150000"]
    end_time_list = ["20231107_164000", "20231107_164000","20231107_164000", "20231107_164000"]
    dir_list = [11, 10, 51, 50]
    is_acc_list = [-1,-1, -1, -1]
    
    processes = []
    
    for i in range(len(start_time_list)):
        p = Process(target=process_radar_data, args=(start_time_list[i], end_time_list[i], dir_list[i], is_acc_list[i]))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
def main_get_inference_tst_dataset():

    start_time_list=["20231107_150000","20231107_150000","20231107_150000","20231107_150000"]
    end_time_list=  ["20231107_164000","20231107_164000","20231107_164000","20231107_164000"]
    dir_list=       [11,10,51,50]
    is_acc_list=    [-1,-1,-1,-1]

    for i in range(len(start_time_list)):
        start_time = datetime.datetime.strptime(start_time_list[i],"%Y%m%d_%H%M%S")
        end_time = datetime.datetime.strptime(end_time_list[i],"%Y%m%d_%H%M%S")
        dir = dir_list[i]
        is_acc = is_acc_list[i]
        radar_feature = Radar_Feature(start_time=start_time,end_time=end_time,path='./data/mmAcc/5008/pkl/')
        radar_feature.cal_feature([dir])
        save_name = "./npy/"+ start_time.strftime('%Y%m%d_%H%M%S')+"_"+end_time.strftime('%Y%m%d_%H%M%S')+ "dir"+str(int(dir//10))+"_"+"in"+str(dir%10)+"_data_"
        if is_acc==1:
            np.save(save_name+"acc.npy",radar_feature.feature_data_dict[dir])
        elif is_acc==0:
            np.save(save_name+"normal.npy",radar_feature.feature_data_dict[dir])
        elif is_acc==-1:
            np.save(save_name+"inference.npy",radar_feature.feature_data_dict[dir])
                    
        np.save(save_name+"time.npy",radar_feature.feature_time_dict[dir])


def main_get_npy_dataset():

    start_time_list=["20231107_151000","20231107_154500","20231107_153500","20231107_150000","20231107_155500","20231107_152500","20231107_152500","20231107_161500"]
    end_time_list=  ["20231107_152500","20231107_171000","20231107_155000","20231107_152500","20231107_160500","20231107_154000","20231107_154000","20231107_163000"]
    dir_list=       [11,                11,               50,               50,               51,               51,               10,               10              ] 
    is_acc_list=    [1,                  0,               1,                0,                1,                0,                0,                 1              ]

    # start_time_list = ["20231106_080000","20231106_080000","20231106_080000","20231106_080000"]
    # end_time_list =   ["20231106_090000","20231106_090000","20231106_090000","20231106_090000"] 
    # dir_list = [ 11,10, 51, 50]
    # is_acc_list = [0,0,0,0]
    
    processes = []
    
    for i in range(len(start_time_list)):
        process_radar_data(start_time_list[i], end_time_list[i], dir_list[i], is_acc_list[i])

    # for i in range(len(start_time_list)):
    # # for i in range(1):
    #     i=1
    #     p = Process(target=process_radar_data, args=(start_time_list[i], end_time_list[i], dir_list[i], is_acc_list[i]))
    #     processes.append(p)
    #     p.start()
    
    # for p in processes:
    #     p.join()


    # for i in range(len(start_time_list)):
    #     start_time = datetime.datetime.strptime(start_time_list[i],"%Y%m%d_%H%M%S")
    #     end_time = datetime.datetime.strptime(end_time_list[i],"%Y%m%d_%H%M%S")
    #     dir = dir_list[i]
    #     is_acc = is_acc_list[i]
    #     radar_feature = Radar_Feature(start_time=start_time,end_time=end_time,path='./data/mmAcc/5008/pkl/')
    #     radar_feature.cal_feature([dir])
    #     save_name = "./npy/"+ start_time.strftime('%Y%m%d_%H%M%S')+"_"+end_time.strftime('%Y%m%d_%H%M%S')+ "dir"+str(int(dir//10))+"_"+"in"+str(dir%10)+"_data_"
    #     if is_acc==1:
    #         np.save(save_name+"acc.npy",radar_feature.feature_data_dict[dir])
    #     elif is_acc==0:
    #         np.save(save_name+"normal.npy",radar_feature.feature_data_dict[dir])
    #     elif is_acc==-1:
    #         np.save(save_name+"inference.npy",radar_feature.feature_data_dict[dir])
                    
    #     np.save(save_name+"time.npy",radar_feature.feature_time_dict[dir])
def update_image_within_N(image, x, y, N,value):
    rows, cols = image.shape
    for xi in range(x - N, x + N + 1):
        for yj in range(y - N, y + N + 1):
            # 确保坐标在图像范围内
            if 0 <= xi < rows and 0 <= yj < cols:
                # 检查是否满足距离条�?
                if abs(xi - x) <= N or abs(yj - y) <= N:
                    image[xi, yj] += value

def update_image(image, x, y):
    # 确保坐标在图像范围内
    rows, cols = image.shape
    image[x, y] += 8
    for i in range(-1, 2):
        for j in range(-1, 2):
            # 计算相邻像素的坐�?
            xi = x + i
            yj = y + j
            # 检查坐标是否在图像边界�?
            if 0 <= xi < rows and 0 <= yj < cols:
                # 对于中心点及其直接相邻点增加 2
                image[xi, yj] += 4
                # 遍历中心点相邻点的相邻点
                for ii in range(-1, 2):
                    for jj in range(-1, 2):
                        # 计算次级相邻像素的坐�?
                        xii = xi + ii
                        yjj = yj + jj
                        # 检查坐标是否在图像边界�?
                        if 0 <= xii < rows and 0 <= yjj < cols:
                            # 为次级相邻点增加 1
                            image[xii, yjj] += 2

def normalize_coordinates(list_x, list_y, X1, Y1):
    min_x, max_x = min(list_x), max(list_x)
    min_y, max_y = min(list_y), max(list_y)

    scale_factor_x = X1 / (max_x - min_x)
    scale_factor_y = Y1 / (max_y - min_y)

    normalized_x = [(x - min_x) * scale_factor_x for x in list_x]
    normalized_y = [(y - min_y) * scale_factor_y for y in list_y]

    return normalized_x, normalized_y
def update_image_within_N_optimized(image, x, y, N, value):
    rows, cols = image.shape
    # 计算更新区域的边�?
    x_min = max(x - N, 0)
    x_max = min(x + N + 1, rows)
    y_min = max(y - N, 0)
    y_max = min(y + N + 1, cols)

    # 创建一个掩码，表示需要更新的区域
    mask = np.zeros_like(image, dtype=bool)
    mask[x_min:x_max, y_min:y_max] = True

    # 使用掩码更新图像
    image[mask] += value

def get_lanecode_list(raw_data,dir,length=10000,th =30):

    dict_lanecode = {}
    if length>len(raw_data):
        length = len(raw_data)
        print("ERROR")
        print("Len of raw_data is "+str(len(raw_data))+" but length is "+str(length))

    for frame in raw_data[0:length]:
        for obj in frame.obj_info:
            if (obj.obj_lanenum<200 and obj.radar_dir == dir//10 and obj.is_in_lane==dir%10):
                # if obj.obj_lanenum <6:
                #     print("ERROR")
                if obj.obj_lanenum not in dict_lanecode:
                    dict_lanecode[obj.obj_lanenum]=1
                else:
                    dict_lanecode[obj.obj_lanenum]+=1
    print(dict_lanecode)
    re_list=[]
    for key in dict_lanecode.keys():
        if dict_lanecode[key]>th:
            re_list.append(key)
            # del dict_lanecode[key]
    print(re_list)
    return re_list

def genetare_image(start_time,end_time,path,dir,label,data_ip,save_dir='./data/img_dataset/',frame_len_min=2,frame_diff_msec=1000):
    start_time = datetime.datetime.strptime(start_time, "%Y%m%d_%H%M%S")
    end_time = datetime.datetime.strptime(end_time, "%Y%m%d_%H%M%S")
    radar_feature = Radar_Feature(start_time=start_time,end_time=end_time,path=path)
    lanecode_list = get_lanecode_list(raw_data=radar_feature.raw_data,dir=dir)

    check_time = start_time
    while True:
        check_time = check_time+datetime.timedelta(milliseconds=frame_diff_msec)
        if frame_len_min<100:
            check_time2_end = check_time+datetime.timedelta(minutes=frame_len_min)
        if check_time>end_time:
            break
        # radar_feature = Radar_Feature(start_time=start_time,end_time=check_time2_end,path=path)
        index_s = bisect.bisect(radar_feature.raw_data_time,check_time)
        if frame_len_min<100:
            index_e = bisect.bisect(radar_feature.raw_data_time,check_time2_end)    

        x_label = []
        y_label = []
        speed =[]
        lane_dict_obj_num = {}

        # x_speed = []
        # y_speed = []      
        start_timet = time.time()  
        frame_cnt =0
        while True:
            if (index_s+frame_cnt) > len(radar_feature.raw_data)-1:
                break
            frame  = radar_feature.raw_data[index_s+frame_cnt]
            frame_cnt +=1
        # for frame in radar_feature.raw_data[index_s:index_e]:
            # obj_info = [obj for obj in frame.obj_info if (obj.radar_dir == dir//10 and obj.obj_lanenum<=200 and obj.is_in_lane==dir%10)]
            for obj in frame.obj_info:
                if not (obj.obj_lanenum in lanecode_list and obj.radar_dir == dir//10 and obj.is_in_lane==dir%10):
                    continue    
                speed.append(obj.speed)
                x_label.append(obj.x)
                y_label.append(obj.y)
                if obj.obj_lanenum not in lane_dict_obj_num:

                    lane_dict_obj_num[obj.obj_lanenum]=[0]*31
                    lane_dict_obj_num[obj.obj_lanenum][(int)(obj.x/1000)]+=1
                else:
                    lane_dict_obj_num[obj.obj_lanenum][(int)(obj.x/1000)]+=1
                    # lane_dict_obj_num[obj.obj_lanenum].append((int)(obj.x/1000))
            if frame_len_min<100:
                if index_e<=index_s+frame_cnt:
                    break
            elif frame_len_min>100:
                if len(x_label)>frame_len_min or len(y_label)>frame_len_min:
                    break
        frame_end_time = frame.time
        if frame_end_time>end_time:
            break
        
        if len(x_label)<=1000 or len(y_label)<=1000:
            continue
        

        normalized_x, normalized_y = normalize_coordinates(x_label, y_label, 360, 360)

        image = np.zeros((360, 360), dtype=np.float16)  
        for i in range(len(normalized_x)):
            x=int(normalized_x[i])
            y=int(normalized_y[i])
            v = abs(speed[i]/100.0)
            update_image_within_N_optimized(image, y, x, 7,v)        

        normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
        normalized_image = normalized_image.astype(np.uint8)
        # x_label_10 = np.array(x_label[:])/1000
        os.makedirs(save_dir,exist_ok=True)
        s_time = check_time.strftime('%Y%m%d_%H%M%S')+ '_' + check_time.strftime('%f')[:3]
        e_time = frame.time.strftime('%Y%m%d_%H%M%S') + '_' + frame.time.strftime('%f')[:3]
        # e_time = check_time2_end.strftime('%Y%m%d_%H%M%S') + '_' + check_time2_end.strftime('%f')[:3]
        print(save_dir+str(label)+'_label_'+'dir_'+str(dir) +'_'+s_time+"_"+e_time+"_"+str(len(x_label))+'_'+data_ip+'_.png')
        plt.imsave(save_dir+str(label)+'_label_'+'dir_'+str(dir) +'_'+s_time+"_"+e_time+"_"+str(len(x_label))+'_'+data_ip+'_.png', normalized_image, cmap='gray')  

def main_generate_image_inference_dataset():
    start_time_list = ["20231107_151500","20231107_150000", "20231107_150000", "20231107_150000"]
    end_time_list = ["20231107_164000", "20231107_164000","20231107_164000", "20231107_164000"]
    dir_list = [11, 10, 51, 50]
    is_acc_list = [2, 2, 2, 2]

    # start_time_list = ["20221118_101000","20221118_101000","20221118_101000", "20221118_101000", "20221118_101000"]
    # end_time_list = ["20221118_141000", "20221118_141000", "20221118_141000","20221118_141000", "20221118_141000"]
    # dir_list = [ 11,50, 30,70,10]
    # is_acc_list = [2,2, 2,2,2]
    
    # processes = []    
    # for i in range(len(start_time_list)):
    #     p = Process(target=genetare_image, args=(start_time_list[i], end_time_list[i],"./data/mmAcc/5008/pkl/", dir_list[i], is_acc_list[i],'./data/img_dataset/inference1/'))
    #     processes.append(p)
    #     p.start()
    
    # for p in processes:
    #     p.join()
        
    for i in range(len(start_time_list)):
        genetare_image(start_time_list[i], end_time_list[i],"./data/mmAcc/5008/pkl/", dir_list[i], is_acc_list[i],save_dir='./data/img_dataset/inference1/')
        
def main_generate_image_dataset():
    start_time_list=["20231107_151000","20231107_154500","20231107_153500","20231107_150000","20231107_155500","20231107_152500","20231107_152500","20231107_161500"]
    end_time_list=  ["20231107_152500","20231107_171000","20231107_155000","20231107_152500","20231107_160500","20231107_154000","20231107_154000","20231107_163000"]
    dir_list=       [11,                11,               50,               50,               51,               51,               10,               10              ] 
    is_acc_list=    [1,                  0,               1,                0,                1,                0,                0,                 1              ]
    for i in range(2,len(start_time_list)):
        genetare_image(start_time_list[i], end_time_list[i],"./data/mmAcc/5008/pkl/", dir_list[i], is_acc_list[i])

def creat_img(start_time,end_time,path,dir,label):
    radar_feature = Radar_Feature(start_time=start_time,end_time=end_time,path=path)
    x_label = []
    y_label = []
    speed =[]
    x_speed = []
    y_speed = []   

    
    for frame in radar_feature.raw_data[radar_feature.index:radar_feature.indexend]:
        obj_info = [obj for obj in frame.obj_info if (obj.radar_dir == dir//10 and obj.obj_lanenum!=255 and obj.is_in_lane==dir%10)]
        obj_x_coords = [obj.x for obj in obj_info]
        obj_y_coords = [obj.y for obj in obj_info]
        obj_speed = [obj.speed for obj in obj_info]
        obj_x_speed = [obj.vX for obj in obj_info]
        obj_y_speed = [obj.vY for obj in obj_info]
        x_speed += obj_x_speed
        y_speed += obj_y_speed
        speed+=obj_speed
        x_label+=obj_x_coords
        y_label+=obj_y_coords

    normalized_x, normalized_y = normalize_coordinates(x_label, y_label, 360, 360)

    image = np.zeros((360, 360), dtype=np.float16)

    for i in range(len(normalized_x)):
        # x = int((x_label[i]-5000)/10)
        # x/=10
        x=int(normalized_x[i])
        # y = (y_label[i]+3000)
        # y/=10
        y=int(normalized_y[i])
        # image[x, y] += 2
        # update_image(image, x, y)
        v = abs(speed[i]/100)
        update_image_within_N(image, y, x, 7,v)
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255

    # 将数据类型转换为uint8
    normalized_image = normalized_image.astype(np.uint8)
    plt.imsave(label+'output_image.png', normalized_image, cmap='gray')
    # plt.imshow(normalized_image, cmap='gray')
    # plt.colorbar()
    # plt.savefig(f"./fig/tst.png")

def origin_data_deal(path):

    dirs = os.listdir(path)
    for dir in dirs:
        dir_path = os.path.join(path,dir)
        print(dir_path)
        read_dat_to_pkl(dir_path)

def get_image_csv_infor(image_csv_infor_path):


    # 初始化空列表用于存储每列的数据
    start_time_list = []
    end_time_list = []
    dir_list = []
    is_acc_list = []
    ip_list = []
    frame_len_min_list = []
    frame_diff_msec_list = []

    # 读取 CSV 文件
    with open(image_csv_infor_path, 'r') as file:
        reader = csv.reader(file)

        # 跳过标题行
        next(reader)

        # 读取每行数据
        for row in reader:
            start_time_list.append(row[0].strip())  # 去除前后空格
            end_time_list.append(row[1].strip())    # 去除前后空格
            dir_list.append(int(row[2].strip()))    # 去除空格并转换为整数
            is_acc_list.append(int(row[3].strip())) # 去除空格并转换为整数
            ip_list.append(row[4].strip())          # 去除前后空格
            frame_len_min_list.append(int(row[5].strip())) # 去除空格并转换为整数
            frame_diff_msec_list.append(int(row[6].strip())) # 去除空格并转换为整数
            


    print("Start Times:", start_time_list)
    print("End Times:", end_time_list)
    print("Dirs:", dir_list)
    print("Is Acc:", is_acc_list)
    print("IPs:", ip_list)   
    print("Frame Len Min:", frame_len_min_list)
    print("Frame Diff mSec:", frame_diff_msec_list)
    return start_time_list,end_time_list,dir_list,is_acc_list,ip_list,frame_len_min_list,frame_diff_msec_list

def creat_image_dataset(save_path,image_csv_infor_path):

    # start_time_list=["20231107_151300","20231107_154500","20231107_153500","20231107_150000","20231107_155500","20231107_152500","20231107_152500","20231107_161500","20231107_160000","20231107_160000","20231107_160000","20231107_160000"]
    # end_time_list=  ["20231107_153000","20231107_171000","20231107_155000","20231107_152500","20231107_160500","20231107_154000","20231107_154000","20231107_163000","20231107_163000","20231107_163000","20231107_163000","20231107_163000"]
    # dir_list=       [11,                11,               50,               50,               51,               51,               10,               10              , 30              , 31              , 70              , 71              ] 
    # is_acc_list=    [1,                  0,               1,                0,                1,                0,                0,                 1              , 0               ,  0              ,  0              ,  0              ]
    # ip_list =       ["37.31.205.161"  , "37.31.205.161" , "37.31.205.161" , "37.31.205.161" , "37.31.205.161" , "37.31.205.161" , "37.31.205.161" , "37.31.205.161" , "37.31.205.161" , "37.31.205.161" , "37.31.205.161" , "37.31.205.161" ]
    
    start_time_list,end_time_list,dir_list,is_acc_list,ip_list,frame_len,fram_diff = get_image_csv_infor(image_csv_infor_path)
    for i in range(len(start_time_list)):
        genetare_image(start_time_list[i], end_time_list[i],"./data/pkl/"+ip_list[i]+"/", dir_list[i], is_acc_list[i],data_ip=ip_list[i],save_dir=save_path,frame_len_min=frame_len[i],frame_diff_msec=fram_diff[i])

    # for i in range(2,len(start_time_list)):
    #     genetare_image(start_time_list[i], end_time_list[i],"./data/mmAcc/5008/pkl/", dir_list[i], is_acc_list[i])





if __name__ == "__main__":

    # read_dat_to_pkl("./data/tst/")   
    # read_dat_to_pkl("./data/origin_radar/37.31.190.252")
    # read_dat_to_pkl("./data/origin_radar/37.31.205.161")
    # read_dat_to_pkl("./data/origin_radar/172.23.204.91")
    # read_dat_to_pkl("./data/origin_radar/172.23.204.95")

    # origin_data_deal('./data/origin_radar/')
    # creat_image_dataset("./data/img_dataset/test_4_0105_1/","./data/img_dataset/test_4_0105.csv")
    # creat_image_dataset("./data/img_dataset/tst/","./data/img_dataset/tst.csv")
    creat_image_dataset("./data/img_dataset/train5/","./data/img_dataset/train5.csv")
    start_time = datetime.datetime.strptime("20240105_08000", "%Y%m%d_%H%M%S")
    end_time = datetime.datetime.strptime("20240105_150000", "%Y%m%d_%H%M%S")
    display_the_origin_radar_data_once(start_time=start_time,end_time=end_time,path="./data/pkl/37.31.190.252/",dir=51)





    # main_generate_image_dataset()
    # # read_dat_to_pkl("./data/mmAcc/5008/")
    # main_generate_image_inference_dataset()
    # None
    # main_get_npy_dataset()
    # main_get_inference_tst_dataset_mp()
    # main_get_npy_dataset()
    
    # read_dat_to_pkl("./data/mmAcc/5008/")

    # check_time = datetime.datetime(2023,11,7,15,12,0)
    # end_time = datetime.datetime(2023,11,7,16,40,30)
    # check_time2_end = check_time+datetime.timedelta(minutes=1)
    # creat_img(check_time,check_time2_end,"./data/mmAcc/5008/pkl/",11,label='tst_acc1_')
    
    # while True:
    #     check_time = check_time+datetime.timedelta(seconds=1)
    #     if check_time>end_time:
    #         break
    #     check_time2_end = check_time+datetime.timedelta(minutes=2)
    #     check_time4_end = check_time+datetime.timedelta(minutes=4)

    #     display_the_history_trace(check_time,check_time2_end,"./data/mmAcc/5008/pkl/",1,label='tst2')
    #     display_the_history_trace(check_time,check_time4_end,"./data/mmAcc/5008/pkl/",1,label='tst4')
    #     display_the_history_trace(check_time,check_time2_end,"./data/mmAcc/5008/pkl/",5,label='tst2')
    #     display_the_history_trace(check_time,check_time4_end,"./data/mmAcc/5008/pkl/",5,label='tst4')
    # # display_the_origin_radar_data1(start_time=start_time,end_time=end_time,path="./data/mmAcc/5008/pkl/")























# if __name__ == "__main__-":

#     # pcatsne_display2([radar_feature4.feature_data_dict[11], radar_feature2.feature_data_dict[11]])

#     None




