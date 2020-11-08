import pandas
import torch
from torch import nn
from torch.utils import data
from matplotlib import pyplot

# Reading the Dataset
train_data = pandas.read_csv('data/kaggle_house/train.csv')
test_data = pandas.read_csv('data/kaggle_house/test.csv')
all_features = pandas.concat([train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]])

# Data Processing
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (
    x - x.mean()) / x.std())  # standardizing the data
all_features[numeric_features] = all_features[numeric_features].fillna(0)
all_features = pandas.get_dummies(all_features, dummy_na=True)  # one-hot encoding

n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float32).view(-1, 1)


# Define the Loss Function
loss = nn.MSELoss()
# Define the Model
net = nn.Sequential(nn.Linear(train_features.shape[-1], 1))
# Define the Error Measure Function
def log_rmse(net, features, labels):
    # stabilize the log value
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rsme = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rsme.item()


# Define the Training Function
def train(train_features, test_features, train_labels, test_labels,
           num_epochs=100, learning_rate=5, weight_decay=0, batch_size=64):

    # Generating Data Sets
    dataset = data.TensorDataset(train_features, train_labels)
    train_iter = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Define the Train Method
    trainer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training
    ls_train, ls_test = [], []
    for i in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y)
            l.backward()
            trainer.step()
            trainer.zero_grad()
        ls_train.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            ls_test.append(log_rmse(net, test_features, test_labels))

    return ls_train, ls_test


# K-fold cross-validation
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    left = i * fold_size
    right = min((i + 1) * fold_size, X.shape[0])
    test_X = X[left:right]
    test_y = y[left:right]
    train_X = torch.cat((X[:left], X[right:]), dim=0)
    train_y = torch.cat((y[:left], y[right:]), dim=0)
    return train_X, test_X, train_y, test_y


def k_fold(k):
    sum_train, sum_test = 0, 0
    for i in range(k):
        k_data = get_k_fold_data(k, i, train_features, train_labels)
        ls_train, ls_test = train(*k_data)
        sum_train += ls_train[-1]
        sum_test += ls_test[-1]
        print(ls_train[-1], ls_test[-1])
    return sum_train / k, sum_test / k

# train_k, test_k = k_fold(k=5)

# Predict
def semilogy(x_vals, y_vals, x_label='epochs', y_label='loss',
             legend=['train', 'test']):
    # pyplot.ion()
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)
    pyplot.semilogy(x_vals, y_vals)
    pyplot.legend(legend)
    pyplot.show()

ls_train, _ = train(train_features, None, train_labels, None)
semilogy(range(100), ls_train)
preds = net(train_features).detach().numpy()
test_data['SalePrice'] = pandas.Series(preds.reshape(len(preds)))
submission = pandas.concat([test_data.Id, test_data.SalePrice], 1)
submission.to_csv('data/kaggle_house/my_submission.csv', index=False)



