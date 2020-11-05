import numpy
import math
import torch
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot

max_degree = 20
n_train, n_test = 100, 100
true_w = torch.zeros(max_degree)
true_w[0:4] = torch.tensor([5, 1.2, -3.4, 5.6])

# Synthetic Data
features = numpy.random.randn(n_train + n_test)
poly_features = numpy.array([numpy.power(features, i) for i in range(max_degree)]).transpose()
features, poly_features = [torch.tensor(x, dtype=torch.float) for x in [features, poly_features]]
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # `gamma(n)` = (n-1)!  avoid very large values for large exponents
labels = (poly_features * true_w).sum(dim=1)
labels += torch.normal(0, 0.1, labels.shape)


def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):

    # Generating Data Sets
    dataset = TensorDataset(train_features, train_labels)
    train_iter = DataLoader(dataset, batch_size=10, shuffle=True)
    # Define the Model
    net = torch.nn.Sequential(torch.nn.Linear(train_features.shape[-1], 1, bias=False))
    # Define the Loss Function
    loss = torch.nn.MSELoss()
    # Define the Train Method
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)

    # Training
    ls_train, ls_test = [], []
    for i in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y.reshape(-1,1))
            l.backward()
            trainer.step()
            trainer.zero_grad()
        ls_test.append((loss(net(test_features), test_labels.reshape(-1, 1))))
        ls_train.append((loss(net(train_features), train_labels.reshape(-1,1))))

    # Plot
    semilogy(range(num_epochs), ls_train, range(num_epochs), ls_test)


def semilogy(x_vals, y_vals, x2_vals, y2_vals, x_label='epochs', y_label='loss',
             legend=['train', 'test']):
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)
    pyplot.semilogy(x_vals, y_vals)
    pyplot.semilogy(x2_vals, y2_vals, linestyle='--')
    pyplot.legend(legend)


# Main
pyplot.subplot(1, 3, 1)
# Third-Order Polynomial Function Fitting
train(poly_features[:n_train, :4], poly_features[n_train:, :4],
      labels[:n_train], labels[n_train:])
pyplot.subplot(1, 3, 2)
# Linear Function Fitting (Under fitting)
train(poly_features[:n_train, :2], poly_features[n_train:, :2],
      labels[:n_train], labels[n_train:])
pyplot.subplot(1, 3, 3)
# Higher-Order Polynomial Function Fitting (Over fitting)
train(poly_features[:n_train, :], poly_features[n_train:, :],
      labels[:n_train], labels[n_train:],num_epochs=1500)
pyplot.show()


