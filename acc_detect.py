from radar_data import RadarFeatureReduction,Radar_Dat,Lane_Info,Obj_Info
import numpy as np
import matplotlib.pyplot as plt
import datetime

# 从pkl中读取数据进行事故检测
def AccidentDetection(start_time, end_time, ip,dir,frame_len=3000):
    pkl_path = "./data/pkl/"+ip+"/"
    start_time = datetime.datetime.strptime(start_time, "%Y%m%d_%H%M%S")
    end_time = datetime.datetime.strptime(end_time, "%Y%m%d_%H%M%S")
    radar_feature = RadarFeatureReduction(start_time=start_time,end_time=end_time,path=pkl_path)
    None




if  __name__ == "__main__":
    start_time = "20240105_080000"
    end_time = "20240105_170000"
    ip = "37.31.190.252"
    AccidentDetection(start_time,end_time,ip,10)
    
    