import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import functional as F
class LSTMBinaryClassifier(nn.Module):
    def __init__(self):
        super(LSTMBinaryClassifier, self).__init__()
        self.hidden_size = 64
        self.num_layers = 2  
        self.lstm = nn.LSTM(24, 64, 2, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        # h0 = torch.zeros(2, 1,64)
        # c0 = torch.zeros(2, 1, 64)
        # out, _ = self.lstm(x, (h0, c0))        
        out, _ = self.lstm(x) 
        
        out = out[:, -1, :]
        out = self.fc(out)
        return out
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 5)  
        self.pool = nn.MaxPool2d(2, 2)  
        self.conv2 = nn.Conv2d(3, 3, 5) 
        self.conv3 = nn.Conv2d(3, 6, 5) 
        self.conv4 = nn.Conv2d(6, 9, 5) 
        self.fc1 = nn.Linear(9*18*18, 16) 
        # self.fc2 = nn.Linear(64, 32)          
        self.fc3 = nn.Linear(16, 2)             

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(x.shape[0], -1)  
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
X_data = torch.rand(1,1,360,360)
device = torch.device("cpu")
input_size = X_data.shape[2]
hidden_size = 64
num_layers = 2
output_size = 2
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('./cnn/best_model.pth'))
onxx_model = torch.onnx.export(model,torch.rand((1,X_data.shape[1],X_data.shape[2],X_data.shape[3])).to(device), './model/onxx_model.onnx', verbose=True)
tracemodel = torch.jit.trace(model,X_data)
tracemodel.save('./model/jit_best_model.pth')