import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random
from sklearn.utils import resample
import datetime
from radar_data import get_npy_dataset,get_npy_feature,Radar_Dat,Lane_Info,Obj_Info, find_files_withend 
import re
from lstm import LSTMBinaryClassifier

if __name__ == '__main__':

    input_size = 24
    hidden_size = 64
    num_layers = 2
    output_size = 2
    model = LSTMBinaryClassifier(input_size, hidden_size, num_layers, output_size).to('cpu')
    model.load_state_dict(torch.load('./model/best_model.pkl',map_location='cpu'))
    scripted_model = torch.jit.script(model)
    scripted_model.save("./model/lstm_radar.pt")
    input_shape = (1, 300,24)
    torch.jit.trace(model,torch.rand(input_shape)).save('jit_model')
