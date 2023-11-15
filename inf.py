import torch
import torch.nn as nn
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
X_data = torch.rand(1,300,24)
device = torch.device("cpu")
input_size = X_data.shape[2]
hidden_size = 64
num_layers = 2
output_size = 2
model = LSTMBinaryClassifier().to(device)

onxx_model = torch.onnx.export(model,torch.rand((1,X_data.shape[1],X_data.shape[2])).to(device), './model/onxx_model.onnx', verbose=True)

tracemodel = torch.jit.trace(model,X_data)
tracemodel.save('./model/jit_best_model.pth')