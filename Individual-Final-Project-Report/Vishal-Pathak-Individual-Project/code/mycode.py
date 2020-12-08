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




#Check 

from Preprocessing import monet_addr,photos_addr
import numpy as np
import torch
from Model import G,dtype,F
from PIL import Image
from torch.autograd import Variable
import matplotlib.pyplot as plt

print(monet_addr[22])
print(photos_addr[22])

def check(img_addr):
    img = Image.open(img_addr)
    img_np = np.zeros((1, 3, 256, 256), dtype=np.float32)
    temp_np = np.asarray(img.resize((256, 256), Image.ANTIALIAS))
    plt.imshow(temp_np)
    plt.draw()
    plt.savefig('temp.png')

    img_np[0] = temp_np.transpose(2, 0, 1)

    img_np /= 255
    img_np = img_np * 2 - 1
    img_tensor = torch.from_numpy(img_np)
    img_var = Variable(img_tensor).type(dtype)

    photo_var = G(img_var)
    photo = photo_var.data.cpu().numpy()
    photo = photo[0].transpose(1, 2, 0)
    photo = (photo + 1) / 2
    plt.figure()
    plt.imshow(photo)
    plt.draw()
    plt.savefig('t1.png')
    #plt.show(photo)

    paint_var = F(photo_var)
    paint = paint_var.data.cpu().numpy()
    paint = paint[0].transpose(1, 2, 0)
    paint = (paint + 1) / 2
    plt.figure()
    plt.imshow(paint)
    plt.draw()
    plt.savefig('t2.png')
#    plt.show(paint)



check(monet_addr[22])


def photomo(photo_addr):
    img = Image.open(photo_addr)
    img_np = np.zeros((1, 3, 256, 256), dtype=np.float32)
    temp_np = np.asarray(img.resize((256, 256), Image.ANTIALIAS))
    plt.imshow(temp_np)

    img_np[0] = temp_np.transpose(2, 0, 1)

    img_np /= 255
    img_np = img_np * 2 - 1
    img_tensor = torch.from_numpy(img_np)
    img_var = Variable(img_tensor).type(dtype)

    paint_var = F(img_var)
    paint = paint_var.data.cpu().numpy()
    paint = paint[0].transpose(1, 2, 0)
    paint = (paint + 1) / 2
    plt.figure()
    plt.imshow(paint)
    plt.draw()
    plt.savefig('pk.png')
