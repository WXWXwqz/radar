import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.nn import functional as F

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
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 90 * 90, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 90 * 90)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 训练函数
def train_model(net, train_loader, criterion, optimizer, num_epochs,test_loader):
    net.to(device)
    for epoch in range(num_epochs):
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
        test_model(net, test_loader)

# 测试函数
def test_model(net, test_loader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on the test set: {100 * correct / total}%')
    if 100 * correct / total > 99.0:
        torch.save(net.state_dict(), './cnn/best_model.pth')

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
            image = Image.open(img_path).convert('L')
            image = transform(image)
            image = image.unsqueeze(0).to(device)
            img_file_s = img_file.split('_')
            img_time = img_file_s[2]+img_file_s[3]+"__"+img_file_s[4]+'_'+img_file_s[5]
            with torch.no_grad():
                outputs = net(image)
                _, predicted = torch.max(outputs.data, 1)
                f.write(f'{img_time}, {predicted.item()}\n')
                print(f'{img_time}, {predicted.item()}')
def main():
    # 数据集路径
    data_dir = './data/img_dataset/'
    images, labels = get_images_and_labels(data_dir)

    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((360, 360)),
        transforms.ToTensor()
    ])

    train_dataset = ImageDataset(train_images, train_labels, transform=transform)
    test_dataset = ImageDataset(test_images, test_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    net = SimpleCNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    train_model(net, train_loader, criterion, optimizer,20,test_loader)
    

if __name__ == "__main__":
    net = SimpleCNN()
    net.load_state_dict(torch.load('./cnn/best_model.pth'))
    inference_test(net=net,folder_path='./data/img_dataset/inference',output_file='./cnn/result.txt')
    # main()
