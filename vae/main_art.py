from __future__ import print_function
import os
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchvision.io import read_image
import numpy as np
import cv2
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='VAE Example')
parser.add_argument('--train_dir', type=str, default=None)
parser.add_argument('--test_dir', type=str, default=None)
parser.add_argument('--width', type=int, default=32)
parser.add_argument('--height', type=int, default=32)
parser.add_argument('--hdim', type=int, default=32)
parser.add_argument('--ldim', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument(
    '--log_interval',
    type=int,
    default=3,
    metavar='N',
    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            annotations_file,
            img_dir,
            transform=None,
            target_transform=None):
        self.img_dir = img_dir
        self.img_paths = sorted(list(os.listdir(self.img_dir)))
        self.num_imgs = len(self.img_paths)
        print(f'{self.num_imgs} in {self.img_dir}')
        self.img_labels = [0] * self.num_imgs
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def _read_image(self, img_path):
        img = cv2.resize(
            cv2.imread(img_path),
            (args.width,
             args.height),
            interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32) / 255.
        return torch.from_numpy(img)

    def __getitem__(self, idx):
        idx = idx % self.num_imgs
        img_path = os.path.join(self.img_dir, self.img_paths[idx])
        image = self._read_image(img_path)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


train_loader = torch.utils.data.DataLoader(
    CustomImageDataset(
        None,
        args.train_dir),
    batch_size=args.batch_size,
    shuffle=True,
    **kwargs)
test_loader = torch.utils.data.DataLoader(
    CustomImageDataset(
        None,
        args.test_dir),
    batch_size=args.batch_size,
    shuffle=True,
    **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.input_size = args.width * args.height * 3
        self.fc1 = nn.Linear(self.input_size, args.hdim)
        self.fc21 = nn.Linear(args.hdim, args.ldim)
        self.fc22 = nn.Linear(args.hdim, args.ldim)
        self.fc3 = nn.Linear(args.ldim, args.hdim)
        self.fc4 = nn.Linear(args.hdim, self.input_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(
        recon_x, x.view(-1, args.width * args.height * 3), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(
                    data.size(0), args.width, args.height, 3)[:n]])
                save_image(
                    comparison.numpy(),
                    2,
                    f'results/reconstruction_{epoch}.png')

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def save_image(sample, nrow, name):
    ncol = len(sample) // nrow
    fig, axes = plt.subplots(nrow, ncol)
    for i, ax in enumerate(axes.flat):
        ax.imshow(sample[i])
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(name)
    plt.close('all')


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(16, args.ldim).to(device)
            sample = model.decode(sample).cpu()
            sample = sample.view(16, args.width, args.height, 3)
            save_image(sample.numpy(), 4, f'results/sample_{epoch}')
