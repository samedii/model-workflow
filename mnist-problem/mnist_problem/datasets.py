import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


def get_mnist_dataset(train):
    return datasets.MNIST(
        'data',
        train=train,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )

def get_dataset(type='train'):
    '''
    Args:
        type (str): 'train', 'validate', 'test'

    Returns:
        TensorDataset: dataset with features and labels
    '''

    ds = get_mnist_dataset(type == 'train')
    if type == 'test':
        return ds

    n_observations = len(ds)
    training_ratio = 0.7
    n_training_observations = int(n_observations*training_ratio)

    torch.manual_seed(0)
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_training_observations, n_observations - n_training_observations])
    if type == 'train':
        return train_ds
    elif type == 'val':
        return val_ds
    else:
        raise Exception('Unknown type provided')
