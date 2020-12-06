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


# Preprocessing data so that every value in the image array is between -1 and 1.

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
