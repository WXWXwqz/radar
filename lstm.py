import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random
import datetime
from radar_data import get_npy_dataset,get_npy_feature,Radar_Dat,Lane_Info,Obj_Info, find_files_withend  # 从radar_data.py中导入get_npy_dataset函数

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
def test(model, test_loader, criterion, device, confidence_threshold=0.7):
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
    # X_test,y_test = get_npy_dataset(['acc.npy', 'normal.npy','acc1.npy','normal1.npy','normal2.npy'])
    npy_list = find_files_withend('./npy/','.npy')
    acc_npy_list = []
    normal_npy_list = []
    time_npy_list = []
    for file in npy_list:
        if 'time' in file:
            time_npy_list.append(file)
        elif 'acc' in file:
            acc_npy_list.append(file)
        elif 'normal' in file:
            normal_npy_list.append(file)



    # x_data,y_data = get_npy_dataset(["20231107_151000_20231107_152500dir1_in1_data_acc.npy","20231107_161000_20231107_162500dir1_in1_data_normal.npy"])
    X_test,y_test = get_npy_dataset(acc_npy_list+normal_npy_list)
    # X_test,y_test = get_npy_dataset(['20231107_151000_20231107_152500dir11_data_acc.npy'])
    # num_samples = X_test.shape[0]
    # random_indices = np.arange(num_samples)
    # np.random.shuffle(random_indices)
    # x_data_shuffled = X_test[random_indices]
    # y_data_shuffled = y_test[random_indices]

    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()
    test_dataset = TensorDataset(X_test, y_test)

    batch_size = 32
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建模型实例，定义损失函数和优化器
    input_size = 14
    hidden_size = 256
    num_layers = 2
    output_size = 2
    model = LSTMBinaryClassifier(input_size, hidden_size, num_layers, output_size).to(device)
    model.load_state_dict(torch.load('./model/best_model.pkl'))
    # model = torch.load('9932_model.pkl')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    test_loss,test_accuracy,false_negative_rate, false_positive_rate= test(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy * 100:.2f}% - 漏报率: {false_negative_rate * 100:.2f}% - 误报率: {false_positive_rate * 100:.2f}%")
    # print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy * 100:.2f}%")

if __name__ == "__main__":
    # start_time = datetime.datetime(2023,10,31,17,5,0)
    # end_time = datetime.datetime(2023,10,31,17,25,0)
    # get_npy_feature(start_time=datetime.datetime(2023,11,1,17,40,0),end_time=datetime.datetime(2023,11,1,17,45,0),save_names='normal4.npy',dir=52)
    main_test()

if __name__ == "__main__-":   
    # x_data,y_data = get_npy_dataset(['acc.npy', 'normal.npy','acc1.npy','normal1.npy','normal3.npy','normal2.npy','acc4.npy','normal4.npy','20231107_151000_20231107_152500dir11_data_acc.npy'])
    npy_list = find_files_withend('./npy/','.npy')
    acc_npy_list = []
    normal_npy_list = []
    time_npy_list = []
    for file in npy_list:
        if 'time' in file:
            time_npy_list.append(file)
        elif 'acc' in file:
            acc_npy_list.append(file)
        elif 'normal' in file:
            normal_npy_list.append(file)



    # x_data,y_data = get_npy_dataset(["20231107_151000_20231107_152500dir1_in1_data_acc.npy","20231107_161000_20231107_162500dir1_in1_data_normal.npy"])
    x_data,y_data = get_npy_dataset(acc_npy_list+normal_npy_list)
    
    time_label1 = np.load('./npy/20231107_151000_20231107_152500dir1_in1_data_time.npy')
    time_label2 = np.load('./npy/20231107_161000_20231107_162500dir1_in1_data_time.npy')


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
    hidden_size = 256
    num_layers = 2
    output_size = 2
    model = LSTMBinaryClassifier(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    min_loss = float('inf')
    loss_threshold = 1e-4  # 设定您认为“明显”下降的损失阈值
    patience = 20  # 设定在停止前等待改善的时期数
    trigger_times = 0  # 这将计算损失改善低于阈值的次数

    # 训练和测试
    num_epochs = 500

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy, false_negative_rate, false_positive_rate = test(model, test_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}] - 训练损失: {train_loss:.4f} - 测试损失: {test_loss:.4f} - 测试精度: {test_accuracy * 100:.2f}% - 漏报率: {false_negative_rate * 100:.2f}% - 误报率: {false_positive_rate * 100:.2f}%")
        
        # 检查损失是否明显下降
        if test_loss + loss_threshold < min_loss:
            min_loss = test_loss
            trigger_times = 0  # 重置触发次数
            
        else:
            trigger_times += 1  # 损失下降不明显，触发次数加一

        # 如果连续几个epoch损失下降不明显，则早停
        if trigger_times >= patience:
            
            # 当有更好的模型时保存
            current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            best_model_name = f"./model/best_model_{current_time}_loss_{min_loss:.4f}_acc_{test_accuracy:.4f}.pkl"
            torch.save(model.state_dict(), best_model_name)
            torch.save(model.state_dict(), "./model/best_model.pkl")  # 保存另一份名为best的模型
            print(f"训练早停，在第 {epoch+1} 个时期停止。")
            break
    # for epoch in range(num_epochs):
    #     train_loss = train(model, train_loader, criterion, optimizer, device)
    #     # test_loss, test_accuracy = test(model, test_loader, criterion, device)
    #     test_loss,test_accuracy,false_negative_rate, false_positive_rate= test(model, test_loader, criterion, device)
    #     print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy * 100:.2f}% - 漏保 Rate: {false_negative_rate * 100:.2f}% - 误报 Rate: {false_positive_rate * 100:.2f}%")
    #     if test_accuracy > 0.99:
    #         torch.save(model.state_dict(), str(int(test_accuracy*10000))+'_model.pkl')
    #         break