import torch
import cv2
import matplotlib.pyplot as plt
from data import data_utils as data
from torchvision.models.mobilenet import mobilenet_v2
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM
import torch.nn.functional as F
from sklearn.manifold import TSNE
from PIL import Image
import torch.nn as nn
import numpy as np


def normalize(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range

def get_tsne_from_dataloader(dataloader, model):
    features = []
    labelss = []
    
    for inputs, labels in dataloader:
        output = model(inputs)
        features.append(output.detach().numpy())
        labelss.append(labels)
    
    features = np.concatenate(features)
    labelss = np.concatenate(labelss)
    tsne = TSNE(n_components = 2).fit_transform(features)

    return tsne, labelss


def main():
    # Load Appropriate Model

    # MobileNetV2
    model = mobilenet_v2(weights = None)
    model.classifier[1] = torch.nn.Linear(in_features = model.last_channel, out_features = 10)
    model.load_state_dict(torch.load("baseline/trained_baseline_model"))

    # ADDA?

    batch_size = 32
    
    svhn_train, svhn_test = data.get_svhn()
    _, mnist_test = data.get_mnist()

    svhn_test_loader = torch.utils.data.DataLoader(svhn_test, batch_size = batch_size, shuffle = False)
    mnist_loader = torch.utils.data.DataLoader(mnist_test, batch_size = batch_size, shuffle = False)

    tsne, labels = get_tsne_from_dataloader(svhn_test_loader, model)

    tx = tsne[:, 0]
    ty = tsne[:, 1]
 
    tx = normalize(tx)
    ty = normalize(ty)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    num_labels = np.unique(labels)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']

    for label in num_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
 
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
 
        ax.scatter(current_tx, current_ty, c=colors[label], label=label)

    ax.legend(loc='best')
 
# finally, show the plot
    plt.show()


if __name__ == '__main__':
    main()