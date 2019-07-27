import shutil
import os
import random
from time import time

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset

import torchvision
import xml.etree.ElementTree as ET
from tqdm import tqdm
from scipy.stats import truncnorm

# from conf import *

IMG_SIZE = (32, 32)
CHANNEL = {'RGB': 3, 'BW': 1}
IMG_CHANNEL = CHANNEL['RGB']
NUM_CLASSES = 10
LATENT_DIM = 100
LEARNING_RATE_G = 0.0003
LEARNING_RATE_D = 0.0001
BETA_1 = 0.5
EPOCHS = 200
# DECAY = LEARNING_RATE/EPOCHS
BATCH_SIZE = 32
DATA_LOCATION = '../input/all-dogs/all-dogs/'
CONV_DEPTHS_G = 32
CONV_DEPTHS_D = 48

class DataGenerator(Dataset):
    def __init__(self, directory, transform=None, n_samples=np.inf):
        self.directory = directory
        self.transform = transform
        self.n_samples = n_samples

        self.samples = self._load_subfolders_images(directory)
        if len(self.samples) == 0:
            raise RuntimeError("Found 0 files in subfolders of: {}".format(directory))

    def _load_subfolders_images(self, root):
        IMG_EXTENSIONS = (
            '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

        def is_valid_file(x):
            return torchvision.datasets.folder.has_file_allowed_extension(x, IMG_EXTENSIONS)

        required_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(64),
            torchvision.transforms.CenterCrop(64),
        ])

        imgs = []
        paths = []
        for root, _, fnames in sorted(os.walk(root)):
            for fname in sorted(fnames)[:min(self.n_samples, 999999999999999)]:
                path = os.path.join(root, fname)
                paths.append(path)

        for path in paths:
            if is_valid_file(path):
                # Load image
                img = torchvision.datasets.folder.default_loader(path)

                # Get bounding boxes
                annotation_basename = os.path.splitext(os.path.basename(path))[0]
                annotation_dirname = next(
                    dirname for dirname in os.listdir('../input/annotation/Annotation/') if
                    dirname.startswith(annotation_basename.split('_')[0]))
                annotation_filename = os.path.join('../input/annotation/Annotation/',
                                                   annotation_dirname, annotation_basename)
                tree = ET.parse(annotation_filename)
                root = tree.getroot()
                objects = root.findall('object')
                for o in objects:
                    bndbox = o.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)

                    w = np.min((xmax - xmin, ymax - ymin))
                    bbox = (xmin, ymin, xmin + w, ymin + w)
                    object_img = required_transforms(img.crop(bbox))
                    # object_img = object_img.resize((64,64), Image.ANTIALIAS)
                    imgs.append(object_img)
        return imgs

    def __getitem__(self, index):
        sample = self.samples[index]

        if self.transform is not None:
            sample = self.transform(sample)

        return np.asarray(sample)

    def __len__(self):
        return len(self.samples)


class PixelwiseNorm(torch.nn.Module):
    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the module
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y


class MinibatchStdDev(torch.nn.Module):
    """
    Minibatch standard deviation layer for the discriminator
    """

    def __init__(self):
        """
        derived class constructor
        """
        super(MinibatchStdDev, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the layer
        :param x: input activation volume
        :param alpha: small number for numerical stability
        :return: y => x appended with standard deviation constant map
        """
        batch_size, _, height, width = x.shape
        # [B x C x H x W] Subtract mean over batch.
        y = x - x.mean(dim=0, keepdim=True)
        # [1 x C x H x W]  Calc standard deviation over batch
        y = torch.sqrt(y.pow(2.).mean(dim=0, keepdim=False) + alpha)

        # [1]  Take average over feature_maps and pixels.
        y = y.mean().view(1, 1, 1, 1)

        # [B x 1 x H x W]  Replicate over group and pixels.
        y = y.repeat(batch_size, 1, height, width)

        # [B x C x H x W]  Append as new feature_map.
        y = torch.cat([x, y], 1)
        # return the computed values:
        return y


class Generator(torch.nn.Module):
    def __init__(self, nz, nfeats, nchannels):
        super(Generator, self).__init__()

        # input is Z, going into a convolution
        self.conv1 = spectral_norm(torch.nn.ConvTranspose2d(nz, nfeats * 8, 4, 1, 0, bias=False))
        # self.bn1 = nn.BatchNorm2d(nfeats * 8)
        # state size. (nfeats*8) x 4 x 4

        self.conv2 = spectral_norm(torch.nn.ConvTranspose2d(nfeats * 8, nfeats * 8, 4, 2, 1, bias=False))
        # self.bn2 = nn.BatchNorm2d(nfeats * 8)
        # state size. (nfeats*8) x 8 x 8

        self.conv3 = spectral_norm(torch.nn.ConvTranspose2d(nfeats * 8, nfeats * 4, 4, 2, 1, bias=False))
        # self.bn3 = nn.BatchNorm2d(nfeats * 4)
        # state size. (nfeats*4) x 16 x 16

        self.conv4 = spectral_norm(torch.nn.ConvTranspose2d(nfeats * 4, nfeats * 2, 4, 2, 1, bias=False))
        # self.bn4 = nn.BatchNorm2d(nfeats * 2)
        # state size. (nfeats * 2) x 32 x 32

        self.conv5 = spectral_norm(torch.nn.ConvTranspose2d(nfeats * 2, nfeats, 4, 2, 1, bias=False))
        # self.bn5 = nn.BatchNorm2d(nfeats)
        # state size. (nfeats) x 64 x 64

        self.conv6 = spectral_norm(torch.nn.ConvTranspose2d(nfeats, nchannels, 3, 1, 1, bias=False))
        # state size. (nchannels) x 64 x 64
        self.pixnorm = PixelwiseNorm()

    def forward(self, x):
        # x = F.leaky_relu(self.bn1(self.conv1(x)))
        # x = F.leaky_relu(self.bn2(self.conv2(x)))
        # x = F.leaky_relu(self.bn3(self.conv3(x)))
        # x = F.leaky_relu(self.bn4(self.conv4(x)))
        # x = F.leaky_relu(self.bn5(self.conv5(x)))
        x = torch.nn.functional.leaky_relu(self.conv1(x))
        x = torch.nn.functional.leaky_relu(self.conv2(x))
        x = self.pixnorm(x)
        x = torch.nn.functional.leaky_relu(self.conv3(x))
        x = self.pixnorm(x)
        x = torch.nn.functional.leaky_relu(self.conv4(x))
        x = self.pixnorm(x)
        x = torch.nn.functional.leaky_relu(self.conv5(x))
        x = self.pixnorm(x)
        x = torch.tanh(self.conv6(x))

        return x


class Discriminator(torch.nn.Module):
    def __init__(self, nchannels, nfeats):
        super(Discriminator, self).__init__()

        # input is (nchannels) x 64 x 64
        self.conv1 = torch.nn.Conv2d(nchannels, nfeats, 4, 2, 1, bias=False)
        # state size. (nfeats) x 32 x 32

        self.conv2 = spectral_norm(torch.nn.Conv2d(nfeats, nfeats * 2, 4, 2, 1, bias=False))
        self.bn2 = torch.nn.BatchNorm2d(nfeats * 2)
        # state size. (nfeats*2) x 16 x 16

        self.conv3 = spectral_norm(torch.nn.Conv2d(nfeats * 2, nfeats * 4, 4, 2, 1, bias=False))
        self.bn3 = torch.nn.BatchNorm2d(nfeats * 4)
        # state size. (nfeats*4) x 8 x 8

        self.conv4 = spectral_norm(torch.nn.Conv2d(nfeats * 4, nfeats * 8, 4, 2, 1, bias=False))
        self.bn4 = torch.nn.MaxPool2d(2)
        # state size. (nfeats*8) x 4 x 4
        self.batch_discriminator = MinibatchStdDev()

        self.conv5 = spectral_norm(torch.nn.Conv2d(nfeats * 8 + 1, 1, 2, 1, 0, bias=False))
        # state size. 1 x 1 x 1

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.conv1(x), 0.2)
        x = torch.nn.functional.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = torch.nn.functional.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = torch.nn.functional.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = self.batch_discriminator(x)
        x = torch.sigmoid(self.conv5(x))
        # x= self.conv5(x)
        return x.view(-1, 1)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def show_generated_img_all():
    gen_z = torch.randn(32, LATENT_DIM, 1, 1, device=device)
    gen_images = netG(gen_z).to("cpu").clone().detach()
    gen_images = gen_images.numpy().transpose(0, 2, 3, 1)
    gen_images = (gen_images + 1.0) / 2.0
    fig = plt.figure(figsize=(25, 16))
    for ii, img in enumerate(gen_images):
        ax = fig.add_subplot(4, 8, ii + 1, xticks=[], yticks=[])
        plt.imshow(img)
    # plt.savefig(filename)


def show_generated_img():
    noise = torch.randn(1, LATENT_DIM, 1, 1, device=device)
    gen_image = netG(noise).to("cpu").clone().detach().squeeze(0)
    gen_image = gen_image.numpy().transpose(1, 2, 0)
    gen_image = ((gen_image + 1.0) / 2.0)
    plt.imshow(gen_image)
    plt.show()


def train_module(epochs):
    step = 0
    start = time()
    for epoch in range(epochs):
        for ii, (real_images) in tqdm(enumerate(train_loader), total=len(train_loader)):
            end = time()
            if (end - start) > 30000:
                break
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            labels = torch.full((batch_size, 1), real_label, device=device) + np.random.uniform(-0.1, 0.1)

            output = netD(real_images)
            errD_real = criterion(output, labels)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
            fake = netG(noise)
            labels.fill_(fake_label) + np.random.uniform(0, 0.2)
            output = netD(fake.detach())
            errD_fake = criterion(output, labels)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            labels.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, labels)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if step % 500 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                      % (epoch + 1, EPOCHS, ii, len(train_loader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                valid_image = netG(fixed_noise)
            step += 1
            lr_schedulerG.step(epoch)
            lr_schedulerD.step(epoch)

        if epoch % 5 == 0:
            show_generated_img()


def truncated_normal(size, threshold=1):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values


if __name__ == '__main__':
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(p=0.1),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data_gen = DataGenerator(DATA_LOCATION, transform=transform, n_samples=25000)

    train_loader = torch.utils.data.DataLoader(data_gen, shuffle=True, batch_size=BATCH_SIZE, num_workers=4)
    netG = Generator(LATENT_DIM, CONV_DEPTHS_G, IMG_CHANNEL).to(device)
    netD = Discriminator(3, CONV_DEPTHS_D).to(device)
    criterion = torch.nn.BCELoss()
    optimizerD = torch.optim.Adam(netD.parameters(), lr=LEARNING_RATE_G, betas=(BETA_1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=LEARNING_RATE_D, betas=(BETA_1, 0.999))
    lr_schedulerG = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizerG,
                                                                         T_0=EPOCHS // 200, eta_min=0.00005)
    lr_schedulerD = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizerD,
                                                                         T_0=EPOCHS // 200, eta_min=0.00005)

    fixed_noise = torch.randn(25, LATENT_DIM, 1, 1, device=device)

    real_label = 0.8
    fake_label = 0.0
    batch_size = train_loader.batch_size
    # criterion = nn.MSELoss()
    train_module(EPOCHS)
    # torch.save(netG.state_dict(), 'generator.pth')
    # torch.save(netD.state_dict(), 'discriminator.pth')
    if not os.path.exists('../output_images'):
        os.mkdir('../output_images')
    im_batch_size = 50
    n_images = 10000
    for i_batch in range(0, n_images, im_batch_size):
        z = truncated_normal((im_batch_size, 100, 1, 1), threshold=1)
        gen_z = torch.from_numpy(z).float().to(device)
        # gen_z = torch.randn(im_batch_size, 100, 1, 1, device=device)
        gen_images = netG(gen_z)
        images = gen_images.to("cpu").clone().detach()
        images = images.numpy().transpose(0, 2, 3, 1)
        for i_image in range(gen_images.size(0)):
            torchvision.utils.save_image((gen_images[i_image, :, :, :] + 1.0) / 2.0,
                                         os.path.join('../output_images', f'image_{i_batch + i_image:05d}.png'))

    shutil.make_archive('images', 'zip', '../output_images')
    show_generated_img_all()
