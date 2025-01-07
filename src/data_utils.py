from torchvision.datasets import MNIST, USPS, SVHN
from torchvision.transforms import Compose, Resize, ToTensor, Grayscale
from torch.utils.data import DataLoader
import os


BATCH_SIZE = 200
IMAGE_RESCALE = 28


def get_data_loaders(dataset: str):
    '''
    Get data from MNIST, USPS, or SVHN and split into train and test set.
    '''

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    transform_MNIST_USPS = Compose([Resize((IMAGE_RESCALE, IMAGE_RESCALE)), ToTensor()])
    transform_SVHN = Compose([Resize((IMAGE_RESCALE, IMAGE_RESCALE)), Grayscale(), ToTensor()])

    if dataset == 'MNIST':
        train_data = MNIST(
            root = os.path.join(data_dir, 'MNIST'),
            train = True,
            transform = transform_MNIST_USPS,
            download = True
            )

        test_data = MNIST(
            root = os.path.join(data_dir, 'MNIST'),
            train = False,
            transform = transform_MNIST_USPS,
            download = True
            )
    elif dataset == 'USPS':
        train_data = USPS(
            root = os.path.join(data_dir, 'USPS'),
            train = True,
            transform = transform_MNIST_USPS,
            download = True
            )

        test_data = USPS(
            root = os.path.join(data_dir, 'USPS'),
            train = False,
            transform = transform_MNIST_USPS,
            download = True
            )
    elif dataset == 'SVHN':
        train_data = SVHN(
            root = os.path.join(data_dir, 'SVHN'),
            split = 'train',                         
            transform = transform_SVHN, 
            download = True
        )

        test_data = SVHN(
            root = os.path.join(data_dir, 'SVHN'),
            split = 'test',                        
            transform = transform_SVHN,
            download = True
        )
    else:
        raise Exception('Dataset must be MNIST, USPS, or SVHN')

    loaders = {
        'train': DataLoader(train_data, 
                            batch_size = BATCH_SIZE,
                            shuffle = True),
        'test': DataLoader(test_data,
                           batch_size = BATCH_SIZE, 
                           shuffle = True),
    }
    return loaders
