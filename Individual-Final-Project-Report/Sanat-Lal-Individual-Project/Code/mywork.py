import numpy as np
from PIL import Image


import zipfile,fnmatch,os
# To unzip
# rootPath = os.getcwd()
# pattern = '*.zip'
# for root, dirs, files in os.walk(rootPath):
#     for filename in fnmatch.filter(files, pattern):
#         print(os.path.join(root, filename))
#         zipfile.ZipFile(os.path.join(root, filename)).extractall(os.path.join(root, os.path.splitext(filename)[0]))



path_m =  '/home/ubuntu/PycharmProjects/FinalProject/gan-getting-started/monet_jpg/'
path_p = '/home/ubuntu/PycharmProjects/FinalProject/gan-getting-started/photo_jpg/'
#img_size = 600

# torch.backends.cudnn.enabled=False
# torch.backends.cudnn.deterministic=True


monet_addr = [path_m+i for i in os.listdir(path_m)]
photos_addr = [path_p+i for i in os.listdir(path_p)]


len(photos_addr)




def training(monet_addr, photos_addr):
    X_train, Y_train = np.zeros((300, 3, 256, 256), dtype=np.float32), np.zeros((7038, 3, 256, 256), dtype=np.float32)

    for i in range(len(monet_addr)):
        temp_np = np.asarray(
            Image.open(monet_addr[i]).resize((256, 256), Image.ANTIALIAS))  # resizing the image to 128x128
        X_train[i] = temp_np.transpose(2, 0, 1)
        X_train[i] /= 255
        X_train[i] = X_train[i] * 2 - 1

    for i in range(len(photos_addr)):
        temp_np = np.asarray(Image.open(photos_addr[i]).resize((256, 256), Image.ANTIALIAS))
        Y_train[i] = temp_np.transpose(2, 0, 1)
        Y_train[i] /= 255
        Y_train[i] = Y_train[i] * 2 - 1

    return X_train, Y_train


X_train, Y_train = training(monet_addr, photos_addr)



# Model 

from Preprocessing import X_train, Y_train, monet_addr,photos_addr
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

X_torch = torch.from_numpy(X_train)                #Creating Tensors which will later be wrapped into variables
Y_torch = torch.from_numpy(Y_train)


class disc_np(nn.Module):
    def __init__(self):
        super(disc_np, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2)
        self.bn5 = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, 512, kernel_size=4, stride=2)
        self.bn6 = nn.BatchNorm2d(512)

        self.conv7 = nn.Conv2d(512, 512, kernel_size=2, stride=2)

        self.head = nn.Linear(512, 1)

    def forward(self, input):
        x = Fn.leaky_relu(self.conv1(input), negative_slope=0.2)
        x = Fn.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x = Fn.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x = Fn.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.2)
        x = Fn.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.2)
        x = Fn.leaky_relu(self.bn6(self.conv6(x)), negative_slope=0.2)
        x = Fn.leaky_relu(self.conv7(x), negative_slope=0.2)

        x = x.view(x.size(0), -1)
        x = self.head(x)

        return torch.sigmoid(x)


#epoch

epochs = 120
batch_size = 16

G.train()
F.train()
Dg.train()
Df.train()

k = 0

for epoch in range(epochs):
    print('Epoch number: {0}'.format(epoch))

    for batch in range(X_torch.size(0) // batch_size):
        if batch % 100 == 0:
            print('**Batch number: {0}**'.format(batch))

        monet_real = X_torch[batch * batch_size: (batch + 1) * batch_size]
        if k != 7038:
            photo_real = Y_torch[k % 7038: (k + 1) % 7038]
        else:
            photo_real = Y_torch[7038]
            photo_real = photo_real[np.newaxis, ...]
        k += 1

        monet_real = Variable(monet_real).type(dtype)
        photo_real = Variable(photo_real).type(dtype)


        photo_fake = G(monet_real)

        scores_real = ptd(Dg, photo_real)
        scores_real_np = Dgnp(photo_real)
        scores_fake = ptd(Dg, photo_fake)
        scores_fake_np = Dgnp(photo_fake)

        label_fake = Variable(torch.zeros(batch_size)).type(dtype)
        label_real = Variable(torch.ones(batch_size)).type(dtype)

        scores_real = (0.8 * scores_real + 0.2 * scores_real_np)
        scores_fake = (0.8 * scores_fake + 0.2 * scores_fake_np)

        loss1 = torch.mean((scores_real - label_real) ** 2)
        loss2 = torch.mean((scores_fake - label_fake) ** 2)

        Dgopt.zero_grad()

        loss_dg = (loss1 + loss2)
        if batch % 100 == 0:
            print('Discriminator G loss: {0}'.format(loss_dg.data))
        loss_dg.backward()

        Dgopt.step()

        # Train G
        photo_fake = G(monet_real)

        scores_fake = ptd(Dg, photo_fake)
        loss_g = torch.mean((scores_fake - label_real) ** 2) + 10 * torch.mean(torch.abs(G(F(photo_real)) - photo_real))
        if batch % 100 == 0:
            print('Generator G loss: {0}'.format(loss_g.data))

        Gopt.zero_grad()
        loss_g.backward()
        Gopt.step()

        # Train GAN F

        monet_fake = F(photo_real)

        scores_real = ptd(Df, monet_real)
        scores_real_np = Dfnp(monet_real)
        scores_fake = ptd(Df, monet_fake)
        scores_fake_np = Dfnp(monet_fake)

        scores_real = (0.8 * scores_real + 0.2 * scores_real_np)
        scores_fake = (0.8 * scores_fake + 0.2 * scores_fake_np)

        loss1 = torch.mean((scores_real - label_real) ** 2)
        loss2 = torch.mean((scores_fake - label_fake) ** 2)

        Dfopt.zero_grad()

        loss_df = (loss1 + loss2)
        if batch % 100 == 0:
            print('Discriminator F loss: {0}'.format(loss_df.data))
        loss_df.backward()

        Dfopt.step()

        # Train F

        monet_fake = F(photo_real)

        scores_fake = ptd(Df, monet_fake)
        loss_f = torch.mean((scores_fake - label_real) ** 2) + 10 * torch.mean(
            torch.abs(F(G(monet_real)) - monet_real))
        if batch % 100 == 0:
            print('Generator F loss: {0}'.format(loss_f.data))

        Fopt.zero_grad()
        loss_f.backward()
        Fopt.step()

