import sys
import os
main_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(main_dir, 'models'))
from models import Enc_With_Classifier
from data_utils import get_data_loaders
import argparse
import torch
from torch import nn
from torch import optim
import numpy as np


def pretrain(num_epochs, learn_rate, loaders, dataset):
    '''
    Pretrains source encoder and classifier before ADDA
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Enc_With_Classifier().to(device)
    model.train()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learn_rate)
    steps = len(loaders['train'])
    losses = np.zeros(steps * num_epochs)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            b_x = images.to(device)
            b_y = labels.to(device)
            optimizer.zero_grad()
            output = model(b_x)
            loss = loss_func(output, b_y)
            losses[epoch*steps+i] = loss.item()
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{steps}], Loss: {loss.item():.4f}')
                
    # Save model
    save_dir = os.path.join(main_dir, 'saved_weights')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.encoder.state_dict(), os.path.join(save_dir, f"src_enc_{dataset}.pt"))
    torch.save(model.classifier.state_dict(), os.path.join(save_dir, f"classifier_{dataset}.pt"))
    with open(os.path.join(save_dir, f"pretrain_loss_{dataset}.npy"), 'wb') as f:
        np.save(f, losses)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="MNIST")
    parser.add_argument("--seed", default="36", type=int)
    parser.add_argument("--num_epochs", default="80", type=int)
    parser.add_argument("--lr", default="0.001", type=float)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    data_loaders = get_data_loaders(args.dataset)
    pretrain(args.num_epochs, args.lr, data_loaders, args.dataset)
