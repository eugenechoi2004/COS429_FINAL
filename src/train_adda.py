### Code adapted from https://github.com/yuhui-zh15/pytorch-adda/tree/master

import sys
import os
main_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(main_dir, 'models'))
from models import CNNEncoder, Discriminator
from data_utils import get_data_loaders
import argparse
import torch
from torch import nn
from torch import optim
import numpy as np
import copy

weights_dir = os.path.join(main_dir, 'saved_weights')


def train_adda(num_epochs, lr_discrim, lr_target, source_loaders, target_loaders, src_enc_path, domain_shift):
    '''
    Train ADDA
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_enc = CNNEncoder()
    src_enc.load_state_dict(torch.load(src_enc_path, map_location=device, weights_only = True))
    src_enc.to(device)
    target_enc = CNNEncoder()
    # Initialize target encoder with source encoder weights
    target_enc.load_state_dict(copy.deepcopy(src_enc.state_dict()))
    target_enc.to(device)
    discrim = Discriminator().to(device)

    loss_func = nn.CrossEntropyLoss()
    discrim_optimizer = optim.Adam(discrim.parameters(), lr=lr_discrim)
    target_optimizer = optim.Adam(target_enc.parameters(), lr=lr_target)

    target_enc.train()
    discrim.train()

    steps = len(source_loaders['train'])
    discrim_loss_list = np.zeros(steps * num_epochs)
    target_enc_loss_list = np.zeros(steps * num_epochs)

    for epoch in range(num_epochs):
        for i, (source_data, target_data) in enumerate(zip(source_loaders['train'], target_loaders['train'])):
            # Train discriminator
            source_image = source_data[0].to(device)
            target_image = target_data[0].to(device)

            discrim_optimizer.zero_grad()

            source_features = src_enc(source_image)
            target_features = target_enc(target_image)
            features_concat = torch.cat((source_features, target_features), 0)

            pred_concat = discrim(features_concat.detach())
            source_label = torch.ones(source_features.size(0)).long().to(device)
            target_label = torch.zeros(target_features.size(0)).long().to(device)
            label_concat = torch.cat((source_label, target_label), 0)

            discrim_loss = loss_func(pred_concat, label_concat)
            discrim_loss.backward()
            discrim_optimizer.step()

            pred_class = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_class == label_concat).float().mean()
            
            # Train target encoder
            target_optimizer.zero_grad()
            discrim_optimizer.zero_grad()

            target_features = target_enc(target_image)
            target_pred = discrim(target_features)
            target_label = torch.ones(target_features.size(0)).long().to(device)
            target_loss = loss_func(target_pred, target_label)
            target_loss.backward()
            target_optimizer.step()

            discrim_loss_list[steps*epoch+i] = discrim_loss.item()
            target_enc_loss_list[steps*epoch+i] = target_loss.item()

            if (i + 1) % 100 == 0:
                print("Epoch [{}/{}] Step [{}/{}]: d_loss={:.5f} g_loss={:.5f} acc={:.5f}".format(epoch + 1, num_epochs, i + 1, steps, discrim_loss.data.item(), target_loss.data.item(), acc.data.item()))

    torch.save(discrim.state_dict(), os.path.join(weights_dir, f"discrim_{domain_shift}.pt"))
    torch.save(target_enc.state_dict(), os.path.join(weights_dir, f"target_enc_{domain_shift}.pt"))
    with open(os.path.join(weights_dir, f"discrim_loss_{domain_shift}.npy"), 'wb') as f:
        np.save(f, discrim_loss_list)
    with open(os.path.join(weights_dir, f"target_loss_{domain_shift}.npy"), 'wb') as f:
        np.save(f, target_enc_loss_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default="36", type=int)
    parser.add_argument("--num_epochs", default="150", type=int)
    parser.add_argument("--lr_discrim", default="0.0001", type=float)
    parser.add_argument("--lr_target", default="0.0001", type=float)
    parser.add_argument("--source")
    parser.add_argument("--target")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    train_adda(args.num_epochs, args.lr_discrim, args.lr_target, get_data_loaders(args.source), get_data_loaders(args.target), os.path.join(weights_dir, f"src_enc_{args.source}.pt"), f"{args.source}-{args.target}")
