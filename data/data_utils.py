# Edited from ApprenticeZ/pytorch_dataloader_example.py
import numpy as np
from torchvision.datasets import MNIST, SVHN
from torchvision import transforms
import torch


def normalize(data_tensor):
   '''re-scale image values to [-1, 1]'''
   return (data_tensor / 255.) * 2. - 1.


def tile_image(image):
   '''duplicate along channel axis'''
   return image.repeat(3,1,1)


def get_mnist():
   transform_list = [transforms.ToTensor(), transforms.Lambda(lambda x: normalize(x))]
   # raw mnist images: [num, 28, 28] range (0, 255) must be resized to match svhn
   mnist_train = MNIST(root="data/<root_path>", train = True, download = True,
       transform=transforms.Compose(transform_list+[
           transforms.ToPILImage(),
           transforms.Resize(32),
           transforms.ToTensor(),
           transforms.Lambda(lambda x: tile_image(x))
           ]))
   mnist_test = MNIST(root="data/<root_path>", train = False,download = True,
       transform=transforms.Compose(transform_list+[
           transforms.ToPILImage(),
           transforms.Resize(32),
           transforms.ToTensor(),
           transforms.Lambda(lambda x: tile_image(x))
           ]))
   return mnist_train, mnist_test


def get_svhn():
   transform_list = [transforms.ToTensor(), transforms.Lambda(lambda x: normalize(x))]
   svhn_train = SVHN(root="data/<root_path>", split='train', download = True,
       transform=transforms.Compose(transform_list))
   svhn_test = SVHN(root="data/<root_path>", split='test', download = True,
       transform=transforms.Compose(transform_list))
  
   return svhn_train, svhn_test