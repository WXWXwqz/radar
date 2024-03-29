from radar_data import RadarFeatureReduction,Radar_Dat,Lane_Info,Obj_Info,get_image_csv_infor
import numpy as np
import matplotlib.pyplot as plt
import datetime
import bisect
import os
import csv
import cv2
from multiprocessing import Pool

def normalize_coordinates(list_x, list_y, X1, Y1):
    min_x, max_x = min(list_x), max(list_x)
    min_y, max_y = min(list_y), max(list_y)

    scale_factor_x = X1 / (max_x - min_x)
    scale_factor_y = Y1 / (max_y - min_y)

    normalized_x = [(x - min_x) * scale_factor_x for x in list_x]
    normalized_y = [(y - min_y) * scale_factor_y for y in list_y]

    return normalized_x, normalized_y

def Max_Min_xy_get(raw_data,lane_code_list):
    x_label = []
    y_label = []
    for frame in raw_data:
        for obj in frame.obj_info:
            if obj.obj_lanenum in lane_code_list:
                x_label.append(obj.x)
                y_label.append(obj.y)
    return np.min(x_label),np.max(x_label),np.min(y_label),np.max(y_label)

def save_matrix_to_csv(matrix, filename,decimals=1):

    matrix = np.around(matrix, decimals=decimals)
    if not filename.endswith('.csv'):
        filename += '.csv'

    # 将矩阵写入CSV文件
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in matrix:
            writer.writerow(row)

def AdaptAccidentHyperUpdate(start_time, end_time, ip,dir,frame_len=7000,frame_diff_msec=1000,th=20):
    pkl_path = "./data/pkl/"+ip+"/"
    
    save_dir = "./detect_pre/"+ip+"/"
    os.makedirs(save_dir,exist_ok=True)
    if os.path.exists(save_dir+'dir_'+str(dir) +'_HyperPara.npz'):
        return


    start_time = datetime.datetime.strptime(start_time, "%Y%m%d_%H%M%S")
    end_time = datetime.datetime.strptime(end_time, "%Y%m%d_%H%M%S")
    radar_feature = RadarFeatureReduction(start_time=start_time,end_time=end_time,path=pkl_path)
    lanecode_list = radar_feature.get_lanecode_list(raw_data=radar_feature.raw_data,dir=dir)
    lanecode_list.sort()
    x_min,x_max,y_min,y_max = Max_Min_xy_get(radar_feature.raw_data[0:10000],lanecode_list)
    


    x_max-=x_min
    y_max-=y_min

    # 归一化到0.5 米
    x_max = x_max/250.0  
    y_max = y_max/50.0
    y_max+=1
    x_max+=1
    print(x_min,x_max,y_min,y_max)
    res_img = np.zeros((round(y_max)+1,round(x_max)+1),dtype=np.int8)
    res_img1 = np.zeros((round(y_max)+1,round(x_max)+1),dtype=np.int64)
    lane_img = np.zeros((len(lanecode_list),round(x_max)+1),dtype=np.float64)
    s_time = start_time    
    lane_img_cnt=0
    while True:
        lane_img_cnt+=1
        index_s = bisect.bisect(radar_feature.raw_data_time,s_time)
        s_time = s_time+datetime.timedelta(milliseconds=frame_diff_msec)
        if s_time>end_time:
            break
        x_label = []
        y_label = []
        lane_num_label = []
        speed =[]
        frame_cnt =0
        while len(x_label)<frame_len:
            if (index_s+frame_cnt) > len(radar_feature.raw_data)-1:
                break
            frame  = radar_feature.raw_data[index_s+frame_cnt]
            frame_cnt +=1
            for obj in frame.obj_info:
                if not (obj.obj_lanenum in lanecode_list and obj.radar_dir == dir//10 and obj.is_in_lane==dir%10):
                    continue    
                speed.append(obj.speed)
                x_label.append(obj.x)
                y_label.append(obj.y)
                lane_num_label.append(obj.obj_lanenum)
        frame_end_time = frame.time
        if frame_end_time>end_time:
            break
        # 画图
        # x -= x_min
        # y -= y_min
        x = (np.array(x_label)-x_min)/250.0
        y = (np.array(y_label-y_min))/50.0




        img = np.zeros((round(y_max)+1,round(x_max)+1),dtype=np.float64)
        
        
        for i in range(len(x)):
            if y[i]>y_max or x[i]>x_max:
                continue
            img[round(y[i]),round(x[i])] += speed[i]/100.0
            lane_img[lanecode_list.index(lane_num_label[i]),round(x[i])]+=(speed[i]/100)
        # normalized_x, normalized_y = normalize_coordinates(x_label, y_label, 360, 360)
        normalized_image = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
        normalized_image = normalized_image.astype(np.uint8)
        n_img_mead = np.mean(normalized_image)
        bin_image = np.where(normalized_image < n_img_mead*1.11177777777777, 0, 1).astype(np.uint8)
        res_img = np.bitwise_or(res_img, bin_image)
        # res_img1+=bin_image
        res_img1 += img.astype(np.int64)
        # normalized_res_img1 = (res_img1 - np.min(res_img1)) / (np.max(res_img1) - np.min(res_img1)) * 255

        
        
        ss_time = s_time.strftime('%Y%m%d_%H%M%S')+ '_' + s_time.strftime('%f')[:3]
        se_time = frame.time.strftime('%Y%m%d_%H%M%S') + '_' + frame.time.strftime('%f')[:3]
        # e_time = check_time2_end.strftime('%Y%m%d_%H%M%S') + '_' + check_time2_end.strftime('%f')[:3]
        # print(save_dir+'dir_'+str(dir) +'_'+ss_time+"_"+se_time+"_"+str(len(x_label))+'_'+ip+'_.png')
        # plt.imsave(save_dir+'dir_'+str(dir) +'_'+ss_time+"_"+se_time+"_"+str(len(x_label))+'_'+ip+'_nor.png', normalized_image, cmap='gray')  
        # plt.imsave(save_dir+'dir_'+str(dir) +'_'+ss_time+"_"+se_time+"_"+str(len(x_label))+'_'+ip+'_bin.png', bin_image, cmap='gray')
        # plt.imsave(save_dir+'dir_'+str(dir) +'_'+ss_time+"_"+se_time+"_"+str(len(x_label))+'_'+ip+'_res.png', res_img, cmap='gray')  
        # plt.imsave(save_dir+'dir_'+str(dir) +'_'+ss_time+"_"+se_time+"_"+str(len(x_label))+'_'+ip+'_res_nor.png', normalized_res_img1, cmap='gray') 
    ss_time = start_time.strftime('%Y%m%d_%H%M%S')+ '_' + start_time.strftime('%f')[:3]
    se_time = end_time.strftime('%Y%m%d_%H%M%S') + '_' + end_time.strftime('%f')[:3]
    mean_res_img = np.mean(res_img1)
    end_bin_image = np.where(res_img1 < mean_res_img*1.11111777777, 0, 1).astype(np.uint8)
    mean_lane_img = np.mean(lane_img)/100
    lane_img/=mean_lane_img
    plt.imsave(save_dir+'dir_'+str(dir) +'_'+ss_time+"_"+se_time+"_"+str(len(x_label))+'_'+ip+'end_bin_image.png', end_bin_image, cmap='gray') 
    save_matrix_to_csv(res_img1, save_dir+'dir_'+str(dir) +'_'+ss_time+"_"+se_time+"_"+str(len(x_label))+'_'+ip+'_res_img1.csv')
    save_matrix_to_csv(end_bin_image, save_dir+'dir_'+str(dir) +'_'+ss_time+"_"+se_time+"_"+str(len(x_label))+'_'+ip+'_end_bin_image.csv')
    save_matrix_to_csv(lane_img.T, save_dir+'dir_'+str(dir) +'_'+ss_time+"_"+se_time+"_"+str(len(x_label))+'_'+ip+'lane_img.csv',decimals=2)
     
    extremaArray = np.array([x_min,x_max,y_min,y_max])
    np.savez(save_dir+'dir_'+str(dir) +'_'+ss_time+"_"+se_time+"_"+str(len(x_label))+'_'+ip+'HyperPara.npz', extremaArray=extremaArray, mask_img=end_bin_image)
    np.savez(save_dir+'dir_'+str(dir) +'_HyperPara.npz', extremaArray=extremaArray, mask_img=end_bin_image,lane_img=lane_img)
        
    return extremaArray,end_bin_image

def preserve_vertical_zeros(matrix, sequence_length=6):
    N, M = matrix.shape
    # 创建一个滑动窗口，长度为sequence_length
    window_sum = np.cumsum(matrix, axis=0)
    window_sum = window_sum[sequence_length-1:] - np.vstack([np.zeros((1, M), dtype=matrix.dtype), window_sum[:-sequence_length]])

    # 检查每列是否存在连续的sequence_length个0
    cols_to_preserve = np.any(window_sum == 0, axis=0)
    
    # 更新矩阵，如果列不满足条件，则将其设置为1
    matrix[:, ~cols_to_preserve] = 1

    return matrix


def AccidentCheckWithHistortyLaneInfor(h_lane_img,cur_lane_img,acc_lane_index,acc_site):


    s_index = acc_site-3
    e_index = acc_site+3
    if s_index<0:
        s_index = 0
    if e_index>len(h_lane_img[acc_lane_index]):
        e_index = len(h_lane_img[acc_lane_index])
    for i in range(s_index,e_index):
        res,cnt = AccidentCheckWithHistortyLaneInfor_(h_lane_img,cur_lane_img,acc_lane_index,i)
        if res:
            return res,cnt
    return res,cnt

def AccidentCheckWithHistortyLaneInfor_(h_lane_img,cur_lane_img,acc_lane_index,acc_site):

    Histort_th = 177
    cur_th = 77
    cnt_th = 17

    h_line = h_lane_img[acc_lane_index]
    c_line = cur_lane_img[acc_lane_index]
    acc_cnt = 0
    # cnt = acc_site  
    s_acc_site = acc_site
    
    cnt =0
    while True:

        if acc_site+cnt<len(h_line) and h_line[acc_site+cnt]>Histort_th and c_line[acc_site+cnt]<cur_th:
            cnt+=1
            # acc_cnt+=1
        else:
            break
    acc_cnt+=cnt
    s_acc_site = acc_site - 1
    cnt =0
    while True:
        if s_acc_site-cnt>0 and  h_line[s_acc_site-cnt]>Histort_th and c_line[acc_site-cnt]<cur_th:
            cnt+=1
        else:
            break
    acc_cnt+=cnt

    if acc_cnt<=cnt_th and acc_cnt >2:
        return True,acc_cnt
    else:
        return False,acc_cnt
        



    # h_lane_img = np.load('./detect_pre/


def AccidentDetection(start_time, end_time, ip,dir,frame_len=3000,frame_diff_msec=1000,dir_name="/home/edgeai/store/radar/detect/"):
    pkl_path = "./data/pkl/"+ip+"/"
    start_time = datetime.datetime.strptime(start_time, "%Y%m%d_%H%M%S")
    end_time = datetime.datetime.strptime(end_time, "%Y%m%d_%H%M%S")
    radar_feature = RadarFeatureReduction(start_time=start_time,end_time=end_time,path=pkl_path)
    lanecode_list = radar_feature.get_lanecode_list(raw_data=radar_feature.raw_data,dir=dir)
    lanecode_list.sort()
    print(lanecode_list)
    s_time = start_time
    data_ = np.load('./detect_pre/'+ip+'/dir_'+str(dir)+'_HyperPara.npz')
    x_min,x_max,y_min,y_max = data_['extremaArray']
    mask_img = data_['mask_img']
    history_lane_img = data_['lane_img']
    mask_img_flip = 1-mask_img
    max_y = y_max
    max_x = x_max
    min_y = y_min
    min_x = x_min

    acc_c1=0
    acc_c2=0    
    cX = 0
    cY = 0
    dict_lane_y_max_min = {}
    while True:
        index_s = bisect.bisect(radar_feature.raw_data_time,s_time)
        s_time = s_time+datetime.timedelta(milliseconds=frame_diff_msec)
        if s_time>end_time:
            break
        x_label = []
        y_label = []
        lane_num_label = []
        speed =[]
        lane_dict_obj_num = {}
        frame_cnt =0
        frame_start_time = None
        dict_lane_y_max_min = {}
        while len(x_label)<frame_len or frame_time_sec<150:
            
            if (index_s+frame_cnt) > len(radar_feature.raw_data)-1:
                break
            frame  = radar_feature.raw_data[index_s+frame_cnt]
            if frame_start_time is None:
                frame_start_time = frame.time

            frame_cnt +=1
            for obj in frame.obj_info:
                if not (obj.obj_lanenum in lanecode_list and obj.radar_dir == dir//10 and obj.is_in_lane==dir%10):
                    continue    
                speed.append(obj.speed)
                x_label.append(obj.x)
                y_label.append(obj.y)
                lane_num_label.append(obj.obj_lanenum)
                if obj.obj_lanenum not in dict_lane_y_max_min:
                    dict_lane_y_max_min[obj.obj_lanenum]=[obj.y,obj.y]
                else:
                    if dict_lane_y_max_min[obj.obj_lanenum][0]<obj.y:
                        dict_lane_y_max_min[obj.obj_lanenum][0] = obj.y
                    if dict_lane_y_max_min[obj.obj_lanenum][1]>obj.y:
                        dict_lane_y_max_min[obj.obj_lanenum][1] = obj.y
            frame_end_time = frame.time
            frame_time_sec = frame_end_time-frame_start_time
            frame_time_sec = frame_time_sec.total_seconds()
            # print(frame_time_sec)
            None

      
        if frame_end_time>end_time:
            break
        # 画图
        # y=y-min_y
        # x=x-min_x       
        x = np.array(x_label-min_x)/250.0
        y = np.array(y_label-min_y)/50.0
        for y_max_min in dict_lane_y_max_min.values():
            y_max_min[0] = (y_max_min[0]-min_y)/50.0
            y_max_min[1] = (y_max_min[1]-min_y)/50.0
            y_max_min[0] = round(y_max_min[0],1)
            y_max_min[1] = round(y_max_min[1],1)
        # min_x = np.min(x)
        # min_y = np.min(y)
        # y=y-min_y
        # x=x-min_x        
        # max_x = np.max(x)
        # max_y = np.max(y)

        img = np.zeros((round(max_y)+1,round(max_x)+1),dtype=np.float64)
        lane_img = np.zeros((len(lanecode_list),round(max_x)+1),dtype=np.float64)
        for i in range(len(x)):
            if y[i]>max_y or x[i]>max_x:
                continue
            img[round(y[i]),round(x[i])] += speed[i]/100
            lane_img[lanecode_list.index(lane_num_label[i]),round(x[i])]+= speed[i]/100
        # normalized_x, normalized_y = normalize_coordinates(x_label, y_label, 360, 360)
        normalized_image = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
        normalized_image = normalized_image.astype(np.uint8)
        n_img_mead = np.mean(normalized_image)
        bin_normalized_image = np.where(normalized_image < 1, 0, 1).astype(np.uint8)
        bin_normalized_image_mask = np.bitwise_or(bin_normalized_image,mask_img_flip)
        kernel = np.ones((2, 2), np.uint8)
        dilated_image = cv2.dilate(bin_normalized_image_mask, kernel, iterations=1)
        eroded_image = cv2.erode(dilated_image, kernel, iterations=1)
        eroded_image_flip = 1-eroded_image
        contours, _ = cv2.findContours(eroded_image_flip, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0

        max_area_list = []
        cX_list=[]
        cY_list=[]
        acc_lane_code_list=[]
        
        # 遍历所有连通区域
        acc_lane =-1
        for contour in contours:
            # 计算每个区域的面积
            area = cv2.contourArea(contour)
            if area>max_y*1.17:
                max_area_list.append(area)                
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                cX_list.append(cX)
                cY_list.append(cY)

                for k in dict_lane_y_max_min.keys():
                    max_y_ ,min_y_ = dict_lane_y_max_min[k]
                    if cY<= max_y_ and cY>=min_y_:
                        acc_lane =k                        
                        break
                acc_lane_code_list.append(acc_lane)

            # 更新最大面积
            if area > max_area:
                if area<max_y*1.17:
                    M = cv2.moments(contour)
                max_area = area   
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])     

        # sorted_lists = sorted(zip(max_area_list, cX_list, cY_list), key=lambda x: x[0], reverse=True)
        # sorted_max_area_list, sorted_cX_list, sorted_cY_list = zip(*sorted_lists)


        acc_res =preserve_vertical_zeros(bin_normalized_image_mask.copy(), sequence_length=5)
        bin_normalized_image_mask[0][0]=0
        acc_res[0][0]=0
        
        
        save_dir = dir_name+ip+"/"+str(dir)+"/"
        save_dir_tmp = dir_name+ip+"/"+str(dir)+"/"+"tmp/"
        result_dir = dir_name+ip+"/"

        os.makedirs(save_dir,exist_ok=True)
        os.makedirs(save_dir_tmp,exist_ok=True)
        ss_time = s_time.strftime('%Y%m%d_%H%M%S')+ '_' + s_time.strftime('%f')[:3]
        se_time = frame.time.strftime('%Y%m%d_%H%M%S') + '_' + frame.time.strftime('%f')[:3]
        # e_time = check_time2_end.strftime('%Y%m%d_%H%M%S') + '_' + check_time2_end.strftime('%f')[:3]
        
        

        print(save_dir+'dir_'+str(dir) +'_'+ss_time+"_"+se_time+"_"+str(len(x_label))+'_'+ip+'_.png')
        # print(dict_lane_y_max_min)
        
        if max_area>max_y*1.17:
            ssss_dir = save_dir
        else:
            ssss_dir = save_dir_tmp

        save_matrix_to_csv(lane_img.T, ssss_dir+'dir_'+str(dir) +'_'+ss_time+"_"+se_time+"_"+str(len(x_label))+'_'+ip+'_laneimg.csv')
        save_matrix_to_csv(normalized_image.T, ssss_dir+'dir_'+str(dir) +'_'+ss_time+"_"+se_time+"_"+str(len(x_label))+'_'+ip+'_bin.csv')
     
        plt.imsave(ssss_dir+'dir_'+str(dir) +'_'+ss_time+"_"+se_time+"_"+str(len(x_label))+'_'+ip+'_nor.png', normalized_image, cmap='gray')  
        # plt.imsave(ssss_dir+'dir_'+str(dir) +'_'+ss_time+"_"+se_time+"_"+str(len(x_label))+'_'+ip+'_bin.png', bin_normalized_image, cmap='gray')  
        # plt.imsave(ssss_dir+'dir_'+str(dir) +'_'+ss_time+"_"+se_time+"_"+str(len(x_label))+'_'+ip+'_bin_mask.png', bin_normalized_image_mask, cmap='gray')  
        # plt.imsave(ssss_dir+'dir_'+str(dir) +'_'+ss_time+"_"+se_time+"_"+str(len(x_label))+'_'+ip+'_acc_res.png', acc_res, cmap='gray')
        plt.imsave(ssss_dir+'dir_'+str(dir) +'_'+ss_time+"_"+se_time+"_"+str(len(x_label))+'_'+ip+'_bin_dilated_image.png', dilated_image, cmap='gray')
        plt.imsave(ssss_dir+'dir_'+str(dir) +'_'+ss_time+"_"+se_time+"_"+str(len(x_label))+'_'+ip+'_bin_eroded_image'+str(max_area)+'.png', eroded_image, cmap='gray')
        acc_lane =-1
        acc_lane_index =-1
        for k in dict_lane_y_max_min.keys():
            max_y_ ,min_y_ = dict_lane_y_max_min[k]
            if cY<= max_y_ and cY>=min_y_:
                acc_lane =k
                break
        
        is_acc =0 
        check_lane_cnt=-1
        if max_area>max_y*1.17:
            for i in range(len(acc_lane_code_list)):
                acc_line_ = acc_lane_code_list[i]
                try:
                    acc_lane_index = lanecode_list.index(acc_line_)
                except:
                    print(cY_list[i],acc_line_,lanecode_list)
                    print(dict_lane_y_max_min)
                    res = False
                    continue
                # acc_lane_index = lanecode_list.index(acc_line_)
                cX = cX_list[i]
                res,check_lane_cnt =  AccidentCheckWithHistortyLaneInfor(h_lane_img=history_lane_img,cur_lane_img=lane_img,acc_lane_index=acc_lane_index,acc_site=cX)
                if res:
                    break    
            if(True==res):
                acc_c2+=1
            else:
                     
                acc_c2=0            
            acc_c1+=1
        else:
            acc_c1=0
            acc_c2=0
        if acc_c1>27:
            is_acc =1
        if acc_c2>27:
            is_acc =2
        
        with open(result_dir+str(dir)+'result.txt', 'a', encoding='utf-8') as file:
            file.write("is_acc,"+str(is_acc)+",acc_c12,["+str(acc_c1)+' '+str(acc_c2) +'],acc_lane,['+str(acc_lane)+' '+str(acc_lane_index)+'],lane_cnt,'+str(check_lane_cnt)+",acc_max_area,"+str(max_area)+',acc_xy ,['+str(cX)+' '+str(cY)+'],img_size,['+str(max_x)+' '+str(max_y)+'],'+'dir,'+str(dir) +','+ss_time+","+se_time+","+str(len(x_label))+','+ip+"\r\n")
    None



def process_task(start_time, end_time, ip, dir, is_acc, frame_len=5777):
    print(start_time, end_time, ip, dir)
    AdaptAccidentHyperUpdate(start_time, end_time, ip, dir, frame_len)
    AccidentDetection(start_time, end_time, ip, dir, frame_len)

def main():
    start_time_list, end_time_list, dir_list, is_acc_list, ip_list, frame_len, fram_diff = get_image_csv_infor('./detect/tst.csv')
    frame_len = 5777  
    num_processes = 10  
    with Pool(processes=num_processes) as pool:
        tasks = [(start_time_list[i], end_time_list[i], ip_list[i], dir_list[i], is_acc_list[i], frame_len) for i in range(len(start_time_list))]
        pool.starmap(process_task, tasks)

if __name__ == "__main_-_":
    main()


if __name__ == "__main_-_":  

    start_time_list,end_time_list,dir_list,is_acc_list,ip_list,frame_len,fram_diff = get_image_csv_infor('./detect/tst.csv')
    for i in range(len(start_time_list)):
        start_time = start_time_list[i]
        end_time = end_time_list[i]
        ip = ip_list[i]
        dir = dir_list[i]
        is_acc = is_acc_list[i]
        print(start_time,end_time,ip,dir)
        AdaptAccidentHyperUpdate(start_time,end_time,ip,dir,frame_len=5777)
        AccidentDetection(start_time,end_time,ip,dir,frame_len=5777)

if  __name__ == "__main__":
    # start_time = "20240105_140000"
    # end_time = "20240105_155900"
    # ip = "37.31.190.252"
    # ip = "172.23.204.91"
    # ip_list=["37.31.190.252"]
    # ip_list = ['37.31.205.161']
    ip_list = ["172.23.204.91"]
    s_time_list = ["20240105_082001"]
    e_time_list = ["20240105_084000"]
    # s_time_list = ["20231218_220000"]
    # e_time_list = ["20231218_235000"]
    adp_start_time_list = ["20240103_090000"]
    adp_end_time_list = ["20240103_095900"]
    dir_list = [71]
    for i in range(len(ip_list)):
        start_time = s_time_list[i]
        end_time = e_time_list[i]
        ip = ip_list[i]
        a_s_time = adp_start_time_list[i]
        a_e_time = adp_end_time_list[i]
        for dir in dir_list:            
            AdaptAccidentHyperUpdate(a_s_time,a_e_time,ip,dir)
            AccidentDetection(start_time,end_time,ip,dir,frame_len=5777,dir_name='/home/edgeai/store/radar/detect_back/'+ip+'_'+str(dir)+"_"+start_time)


    # AdaptAccidentHyperUpdate("20240103_080000","20240103_085900",ip,10)
    # AdaptAccidentHyperUpdate("20240103_080000","20240103_085900",ip,11)
    # AdaptAccidentHyperUpdate("20240103_080000","20240103_085900",ip,30)
    # AdaptAccidentHyperUpdate("20240103_080000","20240103_085900",ip,31)
    # AdaptAccidentHyperUpdate("20240103_080000","20240103_085900",ip,50)
    # AdaptAccidentHyperUpdate("20240103_080000","20240103_085900",ip,51)
    # AdaptAccidentHyperUpdate("20240103_080000","20240103_085900",ip,70)
    # AdaptAccidentHyperUpdate("20240103_080000","20240103_085900",ip,71)
    # # # AccidentDetection(start_time,end_time,ip,50)
    # AccidentDetection(start_time,end_time,ip,10,frame_len=5777)
    # AccidentDetection(start_time,end_time,ip,11,frame_len=5777)
    # AccidentDetection(start_time,end_time,ip,30,frame_len=5777)
    # AccidentDetection(start_time,end_time,ip,31,frame_len=5777)
    # AccidentDetection(start_time,end_time,ip,50,frame_len=5777)
    # AccidentDetection(start_time,end_time,ip,51,frame_len=5777)
    # AccidentDetection(start_time,end_time,ip,70,frame_len=5777)
    # AccidentDetection(start_time,end_time,ip,71,frame_len=5777)
    # # AdaptAccidentHyperUpdate(start_time,end_time,ip,11)  
    
    