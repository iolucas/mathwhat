import math
import os
import numpy as np
from PIL import Image

def load_img_as_array(path):
    img = Image.open(path)
    array = np.array(img)
    return array


def get_int_to_symbol_dict():
    return os.listdir("../data.rar/data/extracted_images")

#int2sym = get_int_to_symbol_dict()
#np.save("int2sym", int2sym)

def load_imgs_as_array(n_batches, batch_index):
    path = "../data.rar/data/extracted_images"
    imgs = []
    labels = []
    for i, symbol in enumerate(os.listdir(path)):
        files = os.listdir("/".join([path, symbol]))
        
        batch_size = math.ceil(len(files)/n_batches)
        #print(batch_size,len(files))
        batch_files = files[batch_index*batch_size:(batch_index + 1)*batch_size]
        
        for file in batch_files:
            labels.append(i)
            img = load_img_as_array("/".join([path, symbol, file]))
            img = img < 128 #Binarize data
            imgs.append(img)
                                    
    return imgs, labels   

def gen_batch_files():
    n_batches = 10
    for batch_ind in range(n_batches):
        print(batch_ind)
        imgs, labels = load_imgs_as_array(n_batches, batch_ind)
        np.save("imgs_batch_{}".format(batch_ind), imgs)
        np.save("labels_batch_{}".format(batch_ind), labels)

def load_dataset(n_batches=10):
    imgs = []
    labels = []
    int2sym = np.load("data/int2sym.npy")
    for batch_i in range(n_batches):
        img_batch = np.load("data/imgs_batch_{}.npy".format(batch_i))
        labels_batch = np.load("data/labels_batch_{}.npy".format(batch_i))
        imgs.append(img_batch)
        labels.append(labels_batch)
        
    return np.concatenate(imgs), np.concatenate(labels), int2sym

