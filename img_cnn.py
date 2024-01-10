import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
import pandas as pd
from collections import Counter
import numpy as np
import time
import random

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载和处理
class ImageDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('L')  # 转换为灰度图像
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_images_and_labels(root_dir):
    images = []
    labels = []

    for img_file in os.listdir(root_dir):
        if img_file.endswith('.png'):
            images.append(os.path.join(root_dir, img_file))
            labels.append(int(img_file[0]))

    return images, labels

# 模型定义
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 5)  # 输入通道1，输出通道6，卷积核大小5
        self.pool = nn.MaxPool2d(2, 2)   # 池化层
        self.conv2 = nn.Conv2d(3, 3, 5) # 输入通道6，输出通道16，卷积核大小5
        self.conv3 = nn.Conv2d(3, 6, 5) # 输入通道6，输出通道16，卷积核大小5
        self.conv4 = nn.Conv2d(6, 9, 5) # 输入通道6，输出通道16，卷积核大小5
        # 减小全连接层的尺寸
        self.fc1 = nn.Linear(9*18*18, 16)  # 第一个全连接层
        # self.fc2 = nn.Linear(64, 32)            # 第二个全连接层
        self.fc3 = nn.Linear(16, 2)             # 第三个全连接层，输出层

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(x.shape[0], -1)  # 展平操作，用于全连接层
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.softmax(x, dim=1)
        return x

# 训练函数
def train_model(net, train_loader, criterion, optimizer, num_epochs,test_loader,exp_name):
    
    for epoch in range(num_epochs):
        net.to(device)
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.2f}%')
        
    tst_acc = test_model(net, test_loader,exp_name)
        # if tst_acc>99.0:
        #     print("stop")
        #     break


def test_model(net, test_loader,exp_name):
    net.eval()
    correct = 0
    total = 0
    confusion_matrix = np.zeros((2, 2))

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            for label, prediction in zip(labels, predicted):
                confusion_matrix[label.item(), prediction.item()] += 1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # 计算每个类别的指标
    TP = confusion_matrix[1, 1]
    FP = confusion_matrix[0, 1]
    TN = confusion_matrix[0, 0]
    FN = confusion_matrix[1, 0]

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    accuracy*=100
    accuracy = round(accuracy,3)
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    recall*=100
    recall = round(recall,3)


    false_discovery_rate = FP / (FP + TP) if FP + TP != 0 else 0
    false_discovery_rate*=100
    false_discovery_rate = round(false_discovery_rate,3)

    print(f"准确率: {accuracy}")
    print(f"召回率: {recall}")
    print(f"误检率: {false_discovery_rate},{FP}/{(FP + TP)}")
    print(f'Accuracy on the test set: {100 * correct / total}%')
    save_dir = "./cnn/"+exp_name+"/"
    os.makedirs(save_dir,exist_ok=True)

    time_str = time.strftime("%Y%m%d%H%M%S", time.localtime())

    save_name = save_dir+time_str+"_acc"+str(accuracy)+"_recall"+str(recall)+"_fdr"+str(false_discovery_rate)+".pth"
    torch.save(net.state_dict(), save_name)

    if 100 * correct / total > 99.0:
        net.to('cpu')
        scripted_model = torch.jit.script(net)

        # 保存脚本化的模型
        # torch.load(python_model, map_location = 'cpu')
        scripted_model.save(save_dir+"script_best_model.pth")
        # input_shape = (1, 300,24)
        input_shape = (1,images.shape[1],images.shape[2],images.shape[3])
        torch.jit.trace(net,torch.rand(input_shape)).save(save_dir+'jit_best_model.pth')
        save_name = save_dir+time_str+"_acc"+str(accuracy)+"_recall"+str(recall)+"_fdr"+str(false_discovery_rate)+"best_model.pth"
        torch.save(net.state_dict(),save_name)
    return 100 * correct / total
def save_tensor_to_csv(tensor, filename):
    # Ensure the tensor is on the CPU and convert to a 2D tensor
    tensor_2d = tensor.cpu().squeeze().numpy()
    
    # Convert to a DataFrame and save as CSV
    df = pd.DataFrame(tensor_2d)
    df.to_csv(filename, index=False)
def inference_test(net, folder_path, output_file):
    net.to(device)
    net.eval()
    
    transform = transforms.Compose([
        transforms.Resize((360, 360)),
        transforms.ToTensor()
    ])

    # 获取文件夹中的所有图片文件名并排序
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    image_files.sort()

    with open(output_file, 'w') as f:
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            save_name = img_file.split('/')[-1]
            save_name =save_name.replace('.png','')
            image = Image.open(img_path).convert('L')
            image = transform(image)
            image = image.unsqueeze(0).to(device)
            img_file_s = img_file.split('_')
            img_time = img_file_s[2]+img_file_s[3]+"__"+img_file_s[4]+'_'+img_file_s[5]
            with torch.no_grad():
                # save_tensor_to_csv(image[0], './cnn/inference.csv')
                # save_tensor_to_csv(image[0]*255, './cnn/inference1.csv')
                outputs = net(image)
                _, predicted = torch.max(outputs.data, 1)
                softmax_out = torch.softmax(outputs, dim=1)
                score = softmax_out[0][1].item()
                if score>0.5:
                    res =1
                else:   
                    res =0
                res_str = "{},{:.2f}".format(res,score*100)
                res_str=save_name+res_str
                res_str=res_str.replace('_',',')
                f.write(res_str+'\n')
                print(res_str)
def train(train_img_dir_list,exp_name):
    # 数据集路径
    # data_dir = './data/img_dataset/'
    images =[]
    labels = []
    for data_dir in train_img_dir_list:
        images_, labels_ = get_images_and_labels(data_dir)
        images.extend(images_)
        labels.extend(labels_)
    # images, labels = get_images_and_labels(data_dir)
    label_counts = Counter(labels)

    # 找到样本数量最少的类别
    min_samples = min(label_counts.values())

    # 欠采样
    balanced_images = []
    balanced_labels = []
    for label in label_counts:
        # 从每个类别中随机选择min_samples个样本
        indices = [i for i, x in enumerate(labels) if x == label]
        chosen_indices = random.sample(indices, min_samples)
        balanced_images.extend([images[i] for i in chosen_indices])
        balanced_labels.extend([labels[i] for i in chosen_indices])
    # train_images, test_images, train_labels, test_labels = train_test_split(
    #     images, labels, test_size=0.3, random_state=42)
    train_images, test_images, train_labels, test_labels = train_test_split(
        balanced_images, balanced_labels, test_size=0.3, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((360, 360)),
        transforms.ToTensor()
    ])
    print(Counter(train_labels))
    train_dataset = ImageDataset(train_images, train_labels, transform=transform)
    test_dataset = ImageDataset(test_images, test_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    net = SimpleCNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    num_epochs = 100
    print("start train")
    for epoch in range(num_epochs):
        net.to(device)
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        step=0
        for images, labels in train_loader:
            step +=1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            step_correct = (predicted == labels).sum().item()
            correct += step_correct
            if step % 100 == 0:
                print(f'step {step}/{len(train_loader)}, step loss: {loss.item():.4f}, step acc: {100*step_correct / len(labels):.2f}%')

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.2f}%')        
        test_model(net, test_loader,exp_name)
    # train_model(net, train_loader, criterion, optimizer,20,test_loader,exp_name)  
def main():
    # 数据集路径
    data_dir = './data/img_dataset/'
    images, labels = get_images_and_labels(data_dir)

    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.3, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((360, 360)),
        transforms.ToTensor()
    ])

    train_dataset = ImageDataset(train_images, train_labels, transform=transform)
    test_dataset = ImageDataset(test_images, test_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    net = SimpleCNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    train_model(net, train_loader, criterion, optimizer,20,test_loader)
    

if __name__ == "__main__":
    net = SimpleCNN()
    # net.load_state_dict(torch.load('./cnn/exp6/20240109131437_acc99.968_recall99.993_fdr0.123best_model.pth'))
    # inference_test(net=net,folder_path='./data/img_dataset/test_4_0105_1',output_file='./cnn/exp6/exp6_test_4_0105_1_result.txt')
    # # main()
    # train(data_dir_list=['./data/img_dataset/train1/','./data/img_dataset/test2/','./data/img_dataset/test1/'],exp_name='exp2')
    train(train_img_dir_list=['./data/img_dataset/train_4_all_time/'],exp_name='exp7')
