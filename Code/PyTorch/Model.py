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


class disc(nn.Module):
    def __init__(self):
        super(disc, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512, 512, kernel_size=2, stride=1)

        self.head = nn.Linear(512, 1)

    def forward(self, input):
        x = Fn.leaky_relu(self.conv1(input), negative_slope=0.2)
        x = Fn.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x = Fn.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x = Fn.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.2)
        x = Fn.leaky_relu(self.conv5(x), negative_slope=0.2)

        x = x.view(x.size(0), -1)
        x = self.head(x)

        return torch.sigmoid(x)


class generator(nn.Module):  # padding concerns: reflection? What exactly is the concept behind convTranspose?

    def __init__(self):
        super(generator, self).__init__()

        # c7s1-32
        self.r1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.bn1 = nn.BatchNorm2d(32)

        # d64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # d128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # R128
        self.r4 = nn.ReflectionPad2d(1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(128)

        self.r5 = nn.ReflectionPad2d(1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(128)

        # R128
        self.r6 = nn.ReflectionPad2d(1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3)
        self.bn6 = nn.BatchNorm2d(128)

        self.r7 = nn.ReflectionPad2d(1)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3)
        self.bn7 = nn.BatchNorm2d(128)

        # R128
        self.r8 = nn.ReflectionPad2d(1)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3)
        self.bn8 = nn.BatchNorm2d(128)

        self.r9 = nn.ReflectionPad2d(1)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3)
        self.bn9 = nn.BatchNorm2d(128)

        # R128
        self.r10 = nn.ReflectionPad2d(1)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=3)
        self.bn10 = nn.BatchNorm2d(128)

        self.r11 = nn.ReflectionPad2d(1)
        self.conv11 = nn.Conv2d(128, 128, kernel_size=3)
        self.bn11 = nn.BatchNorm2d(128)

        # R128
        self.r12 = nn.ReflectionPad2d(1)
        self.conv12 = nn.Conv2d(128, 128, kernel_size=3)
        self.bn12 = nn.BatchNorm2d(128)

        self.r13 = nn.ReflectionPad2d(1)
        self.conv13 = nn.Conv2d(128, 128, kernel_size=3)
        self.bn13 = nn.BatchNorm2d(128)

        # R128
        self.r14 = nn.ReflectionPad2d(1)
        self.conv14 = nn.Conv2d(128, 128, kernel_size=3)
        self.bn14 = nn.BatchNorm2d(128)

        self.r15 = nn.ReflectionPad2d(1)
        self.conv15 = nn.Conv2d(128, 128, kernel_size=3)
        self.bn15 = nn.BatchNorm2d(128)

        # u64
        self.uconv16 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn16 = nn.BatchNorm2d(64)

        # u32
        self.uconv17 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn17 = nn.BatchNorm2d(32)

        # c7s1-3
        self.r18 = nn.ReflectionPad2d(3)
        self.conv18 = nn.Conv2d(32, 3, kernel_size=7, stride=1)
        self.bn18 = nn.BatchNorm2d(3)

    def forward(self, input):
        # c7s1-32
        x = Fn.leaky_relu(self.bn1(self.conv1(self.r1(input))), negative_slope=0.2)

        # d64
        x = Fn.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)

        # d128
        x = Fn.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)

        # R128
        x1 = Fn.leaky_relu(self.bn4(self.conv4(self.r4(x))), negative_slope=0.2)
        x1 = Fn.leaky_relu(self.bn5(self.conv5(self.r5(x1))), negative_slope=0.2)

        x = x + x1

        # R128
        x1 = Fn.leaky_relu(self.bn6(self.conv6(self.r6(x))), negative_slope=0.2)
        x1 = Fn.leaky_relu(self.bn7(self.conv7(self.r7(x1))), negative_slope=0.2)

        x = x + x1

        # R128
        x1 = Fn.leaky_relu(self.bn8(self.conv8(self.r8(x))), negative_slope=0.2)
        x1 = Fn.leaky_relu(self.bn9(self.conv9(self.r9(x1))), negative_slope=0.2)

        x = x + x1

        # R128
        x1 = Fn.leaky_relu(self.bn10(self.conv10(self.r10(x))), negative_slope=0.2)
        x1 = Fn.leaky_relu(self.bn11(self.conv11(self.r11(x1))), negative_slope=0.2)

        x = x + x1

        # R128
        x1 = Fn.leaky_relu(self.bn12(self.conv12(self.r12(x))), negative_slope=0.2)
        x1 = Fn.leaky_relu(self.bn13(self.conv13(self.r13(x1))), negative_slope=0.2)

        x = x + x1

        # R128
        x1 = Fn.leaky_relu(self.bn14(self.conv14(self.r14(x))), negative_slope=0.2)
        x1 = Fn.leaky_relu(self.bn15(self.conv15(self.r15(x1))), negative_slope=0.2)

        x = x + x1

        # u64
        x = Fn.leaky_relu(self.bn16(self.uconv16(x)), negative_slope=0.2)

        # u32
        x = Fn.leaky_relu(self.bn17(self.uconv17(x)), negative_slope=0.2)

        # c7s1-3
        x = Fn.leaky_relu(self.bn18(self.conv18(self.r18(x))), negative_slope=0.2)

        return torch.tanh(x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)


def ptd(discriminator, image):
    score, k = 0, Variable(torch.zeros(1)).type(dtype)
    xp, yp = 0, 0
    x, y = 70, 70
    offset = 25

    while x < 256:
        while y < 256:
            k += 1
            score += discriminator(image[:, :, xp:x, yp:y])
            yp += offset
            y += offset

        xp += offset
        x += offset

    return score / k

dtype = torch.FloatTensor

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor

G = generator().type(dtype)
F = generator().type(dtype)

Dg = disc().type(dtype)
Df = disc().type(dtype)
Dgnp = disc_np().type(dtype)
Dfnp = disc_np().type(dtype)

G.apply(weights_init)
F.apply(weights_init)
Dg.apply(weights_init)
Df.apply(weights_init)

Gopt = optim.Adam(G.parameters(), lr=0.0002)
Fopt = optim.Adam(F.parameters(), lr=0.0002)

Dgopt = optim.Adam(Dg.parameters(), lr=0.0001)
Dfopt = optim.Adam(Df.parameters(), lr=0.0001)


print(Dgnp)



epochs = 3
batch_size = 5

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


