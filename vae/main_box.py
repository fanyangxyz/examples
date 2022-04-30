from __future__ import print_function
import os
import sys

import gflags
import numpy as np
import cv2
from matplotlib import pyplot as plt

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchvision.io import read_image

import draw_box

FLAGS = gflags.FLAGS

gflags.DEFINE_float('lr', '1e-2', '')
gflags.DEFINE_integer('batch_size', 24, '')
gflags.DEFINE_integer('epochs', 30, '')
gflags.DEFINE_integer('log_interval', 100, '')
gflags.DEFINE_boolean('cuda', False, '')
gflags.DEFINE_integer('seed', 33, '')
gflags.DEFINE_integer('ldim', 16, '')
gflags.DEFINE_boolean('verbose', False, '')


BOX_SIZE = 64


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            annotations_file,
            img_dir,
            transform=None,
            target_transform=None):
        self.num_example = 6000

    def __len__(self):
        return self.num_example

    def __getitem__(self, idx):
        image = torch.tensor(draw_box.draw()).float()
        label = 0
        return image, label


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder_model = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=6, kernel_size=3, stride=2), nn.ReLU(), nn.Conv2d(
                in_channels=6, out_channels=12, kernel_size=3, stride=2), nn.ReLU(), nn.Conv2d(
                in_channels=12, out_channels=12, kernel_size=3, stride=2), nn.ReLU(), nn.Conv2d(
                    in_channels=12, out_channels=FLAGS.ldim, kernel_size=3, stride=2), nn.ReLU(), )
        self.fc_mu = nn.LazyLinear(FLAGS.ldim)
        self.fc_logvar = nn.LazyLinear(FLAGS.ldim)

        self.decoder_model = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=FLAGS.ldim,
                out_channels=12,
                kernel_size=3,
                stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=12,
                out_channels=6,
                kernel_size=3,
                stride=3),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=6,
                out_channels=3,
                kernel_size=3,
                stride=3),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=3,
                out_channels=3,
                kernel_size=3,
                stride=3),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x_ncwh = x.permute(0, 3, 1, 2)
        h = self.encoder_model(x_ncwh)
        h_flatten = h.flatten(start_dim=1)
        if FLAGS.verbose:
            print('h flatten')
            print(h_flatten.shape)
        return self.fc_mu(h_flatten), self.fc_logvar(h_flatten)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = z.view(-1, FLAGS.ldim, 1, 1)
        x_hat = self.decoder_model(z)
        x_hat_nwhc = x_hat.permute(0, 2, 3, 1)
        if FLAGS.verbose:
            print('x hat')
            print(x_hat.shape)
        x_cropped = x_hat_nwhc[:, :BOX_SIZE, :BOX_SIZE, :]
        return x_cropped

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    loss = F.binary_cross_entropy(recon_x, x, reduction='mean')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return loss + KLD * 0.5


def save_image(sample, nrow, name):
    assert sample.dtype == 'float32', sample.dtype
    sample = np.clip(sample, 0, 1)
    ncol = len(sample) // nrow
    fig, axes = plt.subplots(nrow, ncol)
    for i, ax in enumerate(axes.flat):
        ax.imshow(sample[i])
        ax.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(name)
    plt.close('all')


def main():
    torch.manual_seed(FLAGS.seed)

    use_cuda = not FLAGS.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        CustomImageDataset(None, None),
        batch_size=FLAGS.batch_size,
        shuffle=True,
        **kwargs)
    test_loader = torch.utils.data.DataLoader(
        CustomImageDataset(None, None),
        batch_size=FLAGS.batch_size,
        shuffle=True,
        **kwargs)

    model = VAE().float().to(device)
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)

    for epoch in range(FLAGS.epochs):
        # training
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(
                recon_batch,
                data,
                mu,
                logvar,
            )
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % FLAGS.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data)))
        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))
        # eval
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                test_loss += loss_function(recon_batch,
                                           data,
                                           mu,
                                           logvar,
                                           )
                if i == 0:
                    # the number of examples to visualize
                    n = min(data.size(0), 8)
                    comparison = torch.cat(
                        [data[:n], recon_batch[:n]])
                    save_image(
                        comparison.numpy(),
                        2,
                        f'results_box/reconstruction_{epoch}.png')
                    np.savez_compressed(
                        f'results_box/reconstruction_{epoch}.npz',
                        comparison.numpy())

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

        # sample
        with torch.no_grad():
            sample = torch.randn(64, FLAGS.ldim).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.numpy(), 8, f'results_box/sample_{epoch}.png')
            np.savez_compressed(
                f'results_box/sample_{epoch}.npz',
                sample.numpy())


if __name__ == "__main__":
    FLAGS(sys.argv)
    main()
