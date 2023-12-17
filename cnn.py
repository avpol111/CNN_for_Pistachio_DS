import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

train_transforms = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_transforms = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_data = ImageFolder(root=os.path.join("Pistachio_CL", "train"),
                         transform=train_transforms)
test_data = ImageFolder(root=os.path.join("Pistachio_CL", "test"),
                         transform=test_transforms)

batch_size = 3
trainloader = DataLoader(train_data, batch_size = batch_size,
                        shuffle = True)

testloader = DataLoader(test_data, batch_size = batch_size,
                        shuffle = True)

class CNN(nn.Module):
  
  def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= 3, out_channels=32, kernel_size= 3)
        self.pool = nn.MaxPool2d(kernel_size= 2, stride= 2)
        
        self.conv2 =  nn.Conv2d(in_channels= 32, out_channels= 64, kernel_size= 3)
        self.conv3 =  nn.Conv2d(in_channels= 64, out_channels= 128, kernel_size= 3)
        self.conv4 =  nn.Conv2d(in_channels= 128, out_channels= 128, kernel_size= 3)
    
        self.fc1 = nn.Linear(128 * 10 * 10, 512)
        self.fc2 = nn.Linear(512, 2)

  def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        x = x.view(-1, 128 * 10 * 10)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

model = CNN()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
n_epochs = 20

for epoch in range(n_epochs):
  running_loss = 0.0
  for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    if i % 100 == 99: # printing the stats every 100 mini-batches
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
        running_loss = 0.0

print("Finished training")

# Let's test:

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct // total} %')

# The accuracy is 93%.