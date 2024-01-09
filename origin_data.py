
def find_dat_files(directory):
    dat_files = []

    # 遍历目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.dat'):
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
                # 文件结束时，如果缓冲区中有数据，将其作为最后一帧存?????
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
                    buffer = buffer[-12:]  # 保留新找到的标识符作为下一个帧的开?????
    
    return frames

def read_dat_file(filename):
    with open(filename, 'rb') as file:
        # 假设每次都读?????4个字节，即一个int值（仅为示例?????
        while True:
            data = file.read(12)  # 读取4个字?????
            if not data:
                break
            # 假设是little-endian的整?????
            str_radar_code  = data.decode()
            value = int.from_bytes(data, byteorder='little')
            # value = str.from
            print(value)

def read_dat_to_pkl(file_path):
    dat_file_list = find_dat_files(file_path)
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
                    pkl_save_path = file_path+'pkl/'
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
        pkl_save_path = file_path+'pkl/'
        os.makedirs(pkl_save_path,exist_ok=True)
        with open(pkl_save_path+last_date+'.pkl', 'wb') as file:
            pickle.dump(pkl_dict[last_date], file)
            print(pkl_save_path+last_date+'.pkl')
            del pkl_dict[last_date] #删除，清理内
