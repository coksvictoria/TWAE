from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from torch.nn.functional import cross_entropy
from torch.optim import Adam
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.autograd import Variable
from twae.util import DataTransformer, reparameterize, _loss_function_MMD,z_gen
# from ttvae.base import BaseSynthesizer, random_state

class Discriminator(nn.Module):
    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(Discriminator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        seq.append(Linear(dim, 1))
        seq.append(nn.Sigmoid())
        self.seq = Sequential(*seq)

    def forward(self,input_):
        x = self.seq(input_)
        return x

class Encoder(nn.Module):
    """Encoder for the TVAE.
    Args:
        data_dim (int):
            Dimensions of the data.
        compress_dims (tuple or list of ints):
            Size of each hidden layer.
        embedding_dim (int):
            Size of the output vector.
    """
    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        self.seq = Sequential(*seq)
        self.fc1 = Linear(dim, embedding_dim)
        self.fc2 = Linear(dim, embedding_dim)

    def forward(self, input_):
        """Encode the passed `input_`."""
        feature = self.seq(input_)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar


class Decoder(nn.Module):
    """Decoder for the TVAE.

    Args:
        embedding_dim (int):
            Size of the input vector.
        decompress_dims (tuple or list of ints):
            Size of each hidden layer.
        data_dim (int):
            Dimensions of the data.
    """

    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)
        self.sigma = Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, input_):
        """Decode the passed `input_`."""
        return self.seq(input_), self.sigma

def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


class TWAE():
    """TWAE."""

    def __init__(
        self,
        l2scale=1e-5,
        batch_size=500,
        epochs=300,
        loss_factor=2,
        latent_dim =32,# Example latent dimension
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        dis_decompress_dims=(128,64,32),
        LAMBDA=0.1,
        embedding_dim=128,
        cuda=True,
        verbose=False,
        device='cuda'
    ):
        self.latent_dim=latent_dim
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.compress_dim=compress_dims
        self.decompress_dims=decompress_dims
        self.embedding_dim=embedding_dim
        self.dis_decompress_dims=dis_decompress_dims
        self.LAMBDA=LAMBDA
        self.loss_factor = loss_factor
        self.epochs = epochs
        self._device = torch.device(device)

    # @random_state
    def fit(self, train_data, discrete_columns=(),save_path=''):
        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)

        self.train_data = self.transformer.transform(train_data).astype('float32')
        dataset = TensorDataset(torch.from_numpy(self.train_data).to(self._device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        data_dim = self.transformer.output_dimensions

        self.encoder = Encoder(data_dim, self.compress_dim, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim).to(self._device)
        discriminator=Discriminator(self.embedding_dim, self.dis_decompress_dims, data_dim).to(self._device)
        dis_optim = Adam(discriminator.parameters(),lr = 0.00005)

        optimizer = Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), weight_decay=self.l2scale)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20, verbose=True)

        self.encoder.train()
        self.decoder.train()
        discriminator.train()

        # one and -one allow us to control descending / ascending gradient descent
        one = torch.tensor(1, dtype=torch.float)

        best_loss = float('inf')
        patience = 0
        start_time = time.time()

        for epoch in range(self.epochs):
        
            pbar = tqdm(loader, total=len(loader))
            pbar.set_description(f"Epoch {epoch+1}/{self.epochs}")

            batch_loss = 0.0
            len_input = 0

            for id_, data in enumerate(pbar):
                optimizer.zero_grad()
                real_x = data[0].to(self._device)
                self.encoder.zero_grad()
                self.decoder.zero_grad()
                discriminator.zero_grad()

                # ======== Train Discriminator ======== #
                frozen_params(self.decoder)
                frozen_params(self.encoder)
                free_params(discriminator)

                z_fake = torch.randn(real_x.size()[0], self.embedding_dim).to(self._device)
                d_fake = discriminator(z_fake)

                mean, std, logvar = self.encoder(real_x)
                z_real = reparameterize(mean, logvar)
                d_real = discriminator(z_real)

                torch.log(d_fake).mean().backward(-one)
                torch.log(1 - d_real).mean().backward(-one)
                dis_optim.step()

                # ======== Train Generator ======== #
                free_params(self.decoder)
                free_params(self.encoder)
                frozen_params(discriminator)

                batch_size = real_x.size()[0]

                mean, std, logvar = self.encoder(real_x)
                z_real = reparameterize(mean, logvar)
                recon_x, sigmas = self.decoder(z_real)
                mean_t, std_t, logvar_t = self.encoder(real_x)
                z_real_t = reparameterize(mean_t, logvar_t)
                d_real = discriminator(z_real_t)

                recon_loss = _loss_function_MMD(recon_x, real_x,sigmas, mean, std,output_info=self.transformer.output_info_list,factor=self.loss_factor)
                d_loss = self.LAMBDA * (torch.log(d_real)).mean()
                loss=recon_loss+d_loss

                batch_loss += loss.item() * len(real_x)
                len_input += len(real_x)

                loss.backward(one)

                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 5)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 5)

                optimizer.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)

                pbar.set_postfix({"Loss": loss.item()})

            curr_loss = batch_loss/len_input
            scheduler.step(curr_loss)

            if curr_loss < best_loss:
              best_loss = loss.item()
              patience = 0
              torch.save(self, save_path+'/model.pt')
            else:
                patience += 1
                if patience == 500:
                    print('Early stopping')
                    break

                        
    # @random_state
    def sample(self, n_samples=100):
        """Sample data similar to the training data.

        """
        self.encoder.eval()
        with torch.no_grad():
            mean, std, logvar = self.encoder(torch.Tensor(self.train_data).to(self._device))

        embeddings = torch.normal(mean=mean, std=std).cpu().detach().numpy()
        synthetic_embeddings=z_gen(embeddings,n_to_sample=n_samples,metric='minkowski',interpolation_method='SMOTE')
        noise = torch.Tensor(synthetic_embeddings).to(self._device)

        self.decoder.eval()
        with torch.no_grad():
          fake, sigmas = self.decoder(noise)
          fake = torch.tanh(fake).cpu().detach().numpy()

        return self.transformer.inverse_transform(fake)

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        self.decoder.to(self._device)