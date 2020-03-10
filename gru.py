import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation
import math, random
from torchvision import datasets, transforms
import dataset
import argparse


# 超参
TIME_STEP = 28 # rnn 时序步长数
INPUT_SIZE = 28 # rnn 的输入维度
BATCH_SIZE = 512
HIDDEN_SIZE = 64 # of rnn 隐藏单元个数
EPOCHS=10 # 总共训练次数
LR = 0.01
h_state = None # 隐藏层状态
train_loss = []  # 误差汇总

class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.gru = nn.GRU(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=2,
            batch_first=True,
        )
        self.out = nn.Linear(HIDDEN_SIZE, 10)  # (hidden_size, output_size)
    
    def forward(self, x):
        out_state, h_state = self.gru(x, None)

        outs = self.out(out_state[:, -1, :])
        # print(outs.shape)
        return outs

def train(args, model, device, train_loader, optimizer, lossfunction, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(-1, 28, 28)
        optimizer.zero_grad()  # 梯度清零
        output = model(data)  # 训练结果
        loss = lossfunction(output, target)  # 计算损失
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新梯度
        train_loss.append(loss.item())
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader, lossfunction):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 28, 28)
            output = model(data)
            test_loss += lossfunction(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    gru = GRU().to(device)
    train_loader, test_loader = dataset.load_mnist(batch_size=BATCH_SIZE, test_batch_size=1000, use_cuda=use_cuda, path="./data")
    optimizer = torch.optim.Adam(gru.parameters(), lr=args.lr) # 优化
    lossFc = nn.CrossEntropyLoss()

    for step in range(EPOCHS):
        train(args, gru, device, train_loader, optimizer, lossFc, step)
    test(args, gru, device, test_loader, lossFc)

    # show loss_line
    x = [i for i in range(len(train_loss))]
    plt.plot(x, train_loss)
    plt.show()

if __name__ == "__main__":
    main()




