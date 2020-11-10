from DataLoader import *
import torch
from torch import nn
from matplotlib import pyplot
import time
# Generating Data Sets
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

# Define the Model
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16*5*5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10)
)
'''
X = torch.randn(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)
'''


# Define the Evaluate Function
def evaluate_accuracy(net, iter):
    acc_sum = 0
    acc_n = 0
    for X, y in iter:
        out = net(X).argmax(dim=1)
        acc_sum += (out == y).float().sum().item()
        acc_n += y.shape[0]
    return acc_sum/acc_n


def semilogy(x_vals, y_vals, x2_vals, y2_vals, x_label='epochs', y_label='accuracy',
             legend=['train', 'test']):
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)
    pyplot.semilogy(x_vals, y_vals)
    pyplot.semilogy(x2_vals, y2_vals, linestyle='--')
    pyplot.legend(legend)
    pyplot.show()


num_epochs = 5
# Define the Loss Function
loss = nn.CrossEntropyLoss()
# Define the Train Method
trainer = torch.optim.Adam(net.parameters(), lr=0.001)

# Training
ls_train, ls_test = [], []
for i in range(num_epochs):
    net.train()
    for X, y in train_iter:
        l = loss(net(X), y)
        l.backward()
        trainer.step()
        trainer.zero_grad()
    net.eval()
    ls_train.append(evaluate_accuracy(net, train_iter))
    ls_test.append(evaluate_accuracy(net, test_iter))
    print(i)
semilogy(range(len(ls_train)), ls_train, range(len(ls_test)), ls_test)
