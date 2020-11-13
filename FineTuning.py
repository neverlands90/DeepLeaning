from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch import nn, optim, device
from torch.utils.data import DataLoader
from matplotlib import pyplot
import time

pretrained_net = models.resnet18(pretrained=True)
pretrained_net.fc = nn.Linear(512, 2)

output_params = list(map(id, pretrained_net.fc.parameters()))
feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())

trainer = optim.SGD([
    {'params': pretrained_net.fc.parameters(), 'lr': 0.1},
    {'params': feature_params}
    ], lr=0.01, weight_decay=0.001)

# 使用ImageNet的均值和标准差
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# 输入数据
train_augs = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
test_augs = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    normalize
])
train_imgs = ImageFolder('./data/hotdog/train', transform=train_augs)
test_imgs = ImageFolder('./data/hotdog/test', transform=test_augs)


# Define the Evaluate Function
def evaluate_accuracy(net, iter):
    acc_sum = 0
    acc_n = 0
    for X, y in iter:
        out = net(X.to(device('cuda'))).argmax(dim=1).cpu()
        acc_sum += (out == y).float().sum().item()
        acc_n += y.shape[0]
    return acc_sum/acc_n


# Define the Plot Function
def myplot(x_vals, y_vals, x2_vals, y2_vals, x_label='epochs', y_label='accuracy',
             legend=['train', 'test']):
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)
    pyplot.plot(x_vals, y_vals)
    pyplot.plot(x2_vals, y2_vals, linestyle='--')
    pyplot.legend(legend)
    pyplot.show()


def train_fine_tuning(net, optimizer, batch_size=64, num_epochs=5):
    train_iter = DataLoader(train_imgs, batch_size, shuffle=True)
    test_iter = DataLoader(test_imgs, batch_size)
    loss = nn.CrossEntropyLoss()

    net = net.to(device('cuda'))


    for i in range(num_epochs):
        time_before = time.time()

        net.train()
        for X, y in train_iter:
            X = X.to(device('cuda'))
            y = y.to(device('cuda'))
            l = loss(net(X), y)
            l.backward()
            trainer.step()
            trainer.zero_grad()

        net.eval()
        ls_train.append(evaluate_accuracy(net, train_iter))
        ls_test.append(evaluate_accuracy(net, test_iter))

        time_after = time.time()
        print('epoch', i, time_after - time_before)


ls_train, ls_test = [], []
train_fine_tuning(pretrained_net, trainer)
myplot(range(len(ls_train)), ls_train, range(len(ls_test)), ls_test)