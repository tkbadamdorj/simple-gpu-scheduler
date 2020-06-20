"""
Author: Taivanbat "TK" Badamdorj
github.com/taivanbat

Adapted from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import torch.optim as optim

import argparse
import os
import json
from glob import glob

from tqdm import tqdm

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__=='__main__':
    # parse hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--log_dir', default="logs")
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    args.cuda = torch.cuda.is_available()

    num_models = len(glob(f'{args.log_dir}/*'))

    # create log directory
    model_log_dir = f'{args.log_dir}/{num_models + 1}'
    if not os.path.exists(model_log_dir):
        os.makedirs(model_log_dir)

    # write hyperparameters to log file
    with open(f'{model_log_dir}/log.txt', 'w') as f:
        f.write(json.dumps(args.__dict__))

    # write hyperparameters to separate file
    with open(f'{model_log_dir}/hparams.json', 'w') as f:
        json.dump(args.__dict__, f)

    # define transforms and dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # define model
    net = Net()

    if args.cuda:
        net = net.cuda()


    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum)

    # train the model
    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader), 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            if args.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


    # test the accuracy on test images
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data

            if args.cuda:
                images, labels = images.cuda(), labels.cuda()

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct/total

    # save accuracy to log file
    with open(f'{model_log_dir}/log.txt', 'a') as f:
        f.write(f'\nfinal accuracy: {accuracy}')

    # write metric to separate file
    with open(f'{model_log_dir}/metrics.json', 'w') as f:
        json.dump({'accuracy': accuracy}, f)

    print('Finished Training')
