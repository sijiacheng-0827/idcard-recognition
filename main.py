import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from torchvision import models


# 加载训练集和测试集
class ImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        lable = self.data.iloc[idx, 1]
        image = Image.open(img_path)
        image = functional.crop(image, left=0, top=0, width=648, height=648)
        if self.transform:
            image = self.transform(image)
        return image, lable

# def to_int16(tensor):
#     return tensor.to(torch.int16)


dataset = ImageDataset(r'D:\card.csv', transform=transforms.Compose([transforms.RandomVerticalFlip(),
                                              transforms.Resize(224),transforms.RandomRotation(10),transforms.ToTensor(),
                                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]))

train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

batch_size = 8
shuffle = True
num_workers = 0

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# 初始化模型、损失函数和优化器
net = models.resnet18(pretrained=True)
for param in net.parameters():
    param.requires_grad = False

net.layer4.requires_grad_ = True
num_classes = 3
net.fc = nn.Linear(net.fc.in_features,num_classes)
# net.lrequires_grad_()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 训练模型
for epoch in range(20):
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print('epoch%d batch%d, loss: %.3f' % (epoch + 1, i + 1, loss))

    if epoch >= 4:
        torch.save(net.state_dict(), './result/idcard_model_{}.pth'.format(epoch + 1))

    print('Epoch: %d,   Loss: %.3f' % (epoch + 1, running_loss))

print('Finished Training')

