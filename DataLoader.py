from torch.utils import data
import torchvision


def load_data_fashion_mnist(batch_size, resize=None):
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(resize))
    trans.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root='/data', train=True, transform=transform, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='/data', train=False, transform=transform, download=True)
    return (data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True),
            data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False))


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[i] for i in labels]
