import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random

# 定义LSTM模型
class LSTMBinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMBinaryClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化LSTM的隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 取LSTM输出的最后一个时间步
        out = out[:, -1, :]

        # 全连接层
        out = self.fc(out)

        return out
# 训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)
def test(model, test_loader, criterion, device, confidence_threshold=0.9):
    model.eval()
    total_loss = 0.0
    correct = 0
    false_negatives = 0
    false_positives = 0
    true_negatives = 0
    true_positives = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # 计算预测的概率
            probabilities = torch.softmax(outputs, dim=1)
            confident_predictions = probabilities[:, 1] > confidence_threshold

            # 计算混淆矩阵中的各项
            for predicted, actual in zip(confident_predictions, labels):
                if actual == 1:
                    if predicted == 1:
                        true_positives += 1
                    else:
                        false_negatives += 1
                else:
                    if predicted == 1:
                        false_positives += 1
                    else:
                        true_negatives += 1

            # 累计正确分类的样本数
            correct += (confident_predictions == labels).sum().item()

    accuracy = correct / len(test_loader.dataset)

    # 计算漏报率和误报率

    if false_negatives == 0:
        false_negative_rate = 0
    else:
        false_negative_rate = false_negatives / (false_negatives + true_positives)
    if false_positives == 0:
        false_positive_rate = 0
    else:
        false_positive_rate = false_positives / (false_positives + true_negatives)

    return total_loss / len(test_loader), accuracy, false_negative_rate, false_positive_rate


# def test(model, test_loader, criterion, device):
#     model.eval()
#     total_loss = 0.0
#     correct = 0
#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             total_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             correct += (predicted == labels).sum().item()

#     accuracy = correct / len(test_loader.dataset)
#     return total_loss / len(test_loader), accuracy

def main_test():
    X_test = np.load('acc1.npy')
    y_test = np.zeros(X_test.shape[0])
    num_samples = X_test.shape[0]
    random_indices = np.arange(num_samples)
    np.random.shuffle(random_indices)

    x_data_shuffled = X_test[random_indices]
    y_data_shuffled = y_test[random_indices]

    X_test = torch.from_numpy(x_data_shuffled).float()
    y_test = torch.from_numpy(y_data_shuffled).long()
    test_dataset = TensorDataset(X_test, y_test)

    batch_size = 32
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建模型实例，定义损失函数和优化器
    input_size = 14
    hidden_size = 64
    num_layers = 2
    output_size = 2
    model = LSTMBinaryClassifier(input_size, hidden_size, num_layers, output_size).to(device)
    model.load_state_dict(torch.load('9932_model.pkl'))
    # model = torch.load('9932_model.pkl')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    test_loss,test_accuracy,false_negative_rate, false_positive_rate= test(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy * 100:.2f}% - False Negative Rate: {false_negative_rate * 100:.2f}% - False Positive Rate: {false_positive_rate * 100:.2f}%")
    # print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy * 100:.2f}%")

if __name__ == "__main__":
    main_test()

if __name__ == "__main__":

    dateset_list = ['acc.npy', 'normal.npy','acc1.npy','normal1.npy','normal2.npy']
    x_data = None
    for datename in dateset_list:
        X_train = np.load(datename)
        if 'acc' in datename:
            y_train = np.ones(X_train.shape[0])
        elif 'normal' in datename:
            y_train = np.zeros(X_train.shape[0])
        if x_data is None:
            x_data = X_train
            y_data = y_train
        else:
            x_data = np.concatenate((x_data, X_train), axis=0)
            y_data = np.concatenate((y_data, y_train), axis=0)
        

    num_samples = x_data.shape[0]
    random_indices = np.arange(num_samples)
    np.random.shuffle(random_indices)

    x_data_shuffled = x_data[random_indices]
    y_data_shuffled = y_data[random_indices]

    X_train = x_data_shuffled[:int(x_data.shape[0]*0.8)]
    y_train = y_data_shuffled[:int(y_data.shape[0]*0.8)]
    X_test = x_data_shuffled[int(x_data.shape[0]*0.8):]
    y_test = y_data_shuffled[int(y_data.shape[0]*0.8):]

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()
    train_dataset = TensorDataset( X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # 创建DataLoader对象用于批处理数据
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建模型实例，定义损失函数和优化器
    input_size = 14
    hidden_size = 64
    num_layers = 2
    output_size = 2
    model = LSTMBinaryClassifier(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 训练和测试
    num_epochs = 500
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        # test_loss, test_accuracy = test(model, test_loader, criterion, device)
        test_loss,test_accuracy,false_negative_rate, false_positive_rate= test(model, test_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy * 100:.2f}% - False Negative Rate: {false_negative_rate * 100:.2f}% - False Positive Rate: {false_positive_rate * 100:.2f}%")
        if test_accuracy > 0.99:
            torch.save(model.state_dict(), str(int(test_accuracy*10000))+'_model.pkl')
            break