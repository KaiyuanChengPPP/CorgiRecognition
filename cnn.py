import sys
import os
import numpy as np
import glob
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
from PIL import Image
import torchvision
import pathlib

class convNet(nn.Module):
    def __init__(self,num_classes = 2):
        super(convNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(num_features = 12)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=24)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(in_features=24*75*75,out_features=num_classes)

    def forward(self,input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = output.view(-1,24*75*75)
        output = self.fc(output)
        return output
classes = ['cardigan', 'pembroke']
def predict(image_path,transformer,model):
    image = Image.open(image_path)
    image_tensor = transformer(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    image_tensor.cpu()
    input = Variable(image_tensor)
    output = model(input)
    index = output.data.numpy().argmax()
    prediction = classes[index]
    print(prediction)
    return prediction

def main():
    train_path = "./data/training_data"
    test_path = "./data/testing_data"
    device = torch.device("cpu")
    print(device)
    transformer = transforms.Compose([
        transforms.Resize((150,150)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],
                             [0.5,0.5,0.5])
    ])
    train_loader = DataLoader(
        torchvision.datasets.ImageFolder(train_path,transform = transformer),batch_size=40,shuffle =True
    )
    test_loader = DataLoader(
        torchvision.datasets.ImageFolder(test_path, transform=transformer), batch_size=20, shuffle=True
    )

    model = convNet(num_classes=2)
    optimizer = Adam(model.parameters(),lr = 0.001, weight_decay = 0.0001)
    loss_function = nn.CrossEntropyLoss()
    num_epochs = 10
    train_count = len(glob.glob(train_path+'/**/*.jpg'))
    test_count = len(glob.glob(test_path + '/**/*.jpg'))
    print(train_count,test_count)

    best_accuracy = 0.0
    best_test_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_accuracy = 0.0
        train_loss = 0.0
        test_loss = 0.0
        test_accuracy = 0.0
        for i, (images,labels) in enumerate(train_loader):
            images = Variable(images.cpu())
            labels = Variable(labels.cpu())
            optimizer.zero_grad()
            output = model(images)
            loss = loss_function(output,labels)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            _,predicted = torch.max(output.data,1)
            train_accuracy+=int(torch.sum(predicted==labels.data))
        train_accuracy = train_accuracy/train_count
        train_loss = train_loss/train_count
        model.eval()
        print('Epoch: ' ,str(epoch) , ' Train Loss ' , str(int(train_loss)) , ' Train Accuracy ' , str(train_accuracy))
        if train_accuracy>best_accuracy:
            torch.save(model.state_dict(),'train.model')
            best_accuracy = train_accuracy

        for i, (images, labels) in enumerate(test_loader):
            images = Variable(images.cpu())
            labels = Variable(labels.cpu())
            optimizer.zero_grad()
            output = model(images)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()
            test_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            test_accuracy += int(torch.sum(predicted == labels.data))
        test_accuracy = test_accuracy / test_count
        test_loss = test_loss / test_count
        model.eval()
        print('Epoch: ', str(epoch), ' Test Loss ', str(int(test_loss)) ,' Test Accuracy ', str(test_accuracy))
        if test_accuracy > best_test_accuracy:
            torch.save(model.state_dict(), 'test.model')
            best_test_accuracy = test_accuracy

    train_model = torch.load('train.model')
    model.load_state_dict(train_model)
    train_car_path = glob.glob(train_path+'/cardigan/*.jpg')
    train_pem_path = glob.glob(train_path + '/pembroke/*.jpg')
    print("Predicting Train Cardigan Folder: ")
    a1,b1,c1,d1 = 0,0,0,0
    a2, b2, c2, d2 = 0, 0, 0, 0
    for i in train_car_path:
        if (predict(i,transformer,model) == "cardigan"):
            a1 = a1 + 1
        a2 = a1 + 1
    print("The accuracy of predicting cardigans in the train data is: ", a1/a2)


    print("Predicting Train Pembroke Folder: ")
    for i in train_pem_path:
        if (predict(i, transformer, model) == "pembroke"):
            b1 = b1 + 1
        b2 = b2 + 1
    print("The accuracy of predicting pembroke in the train data is: ", b1 / b2)

    test_model = torch.load('test.model')
    model.load_state_dict(test_model)
    test_car_path = glob.glob(test_path + '/cardigan/*.jpg')
    test_pem_path = glob.glob(test_path + '/pembroke/*.jpg')
    print("Predicting Test Cardigan Folder: ")
    for i in test_car_path:
        if (predict(i, transformer, model) == "cardigan"):
            c1 = c1 + 1
        c2 = c2 + 1
    print("The accuracy of predicting cardigans in the train data is: ", c1 / c2)
    print("Predicting Test Pembroke Folder: ")
    for i in test_pem_path:
        if (predict(i, transformer, model) == "pembroke"):
            d1 = d1 + 1
        d2 = d2 + 1
    print("The accuracy of predicting pembroke in the train data is: ", d1 / d2)



if __name__ == '__main__':
        main()




