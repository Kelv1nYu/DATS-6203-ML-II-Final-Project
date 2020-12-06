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
