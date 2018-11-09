#-*- coding:utf8 -*-
"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import os
import glob
import h5py
import random
import matplotlib.pyplot as plt
import math
from PIL import Image  # for loading images as YCbCr format
import scipy.misc
import scipy.ndimage
import numpy as np

import tensorflow as tf
#add module
import scipy.io
import cv2
import pandas as pd
import random as rd

FLAGS = tf.app.flags.FLAGS

def read_data(path):
    """
    Read h5 format data file

    Args:
    path: file path of desired file
    data: '.h5' file format that contains train data values
    label: '.h5' file format that contains train label values
    """
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))#h5文件就好比是一个字典，里面含有key，这里的'data'就是key
        label = np.array(hf.get('label'))
        return data, label
def preprocessd_train(path,i):


    image_3 = Image.open(path)
    image_3 = image_3.convert("YCbCr")
    image,cb,cr = image_3.split()
    image = np.array(image,'f')
    cb = np.array(cb,'f')
    cr = np.array(cr,'f')

    image_= image / 255.
    label_ =image / 255.
    gus = scipy.io.loadmat('kernel_10000_21.mat')
    gus = np.array(gus['kernel_10000_21'])
    # gus = scipy.io.loadmat('kernel21_12.mat')
    # gus=np.array(gus['kernel21'])

    # idx = i % 1000

    # idx = j*1000+i
    # kernel = gus[:,:,i]

    if i < 4000:
        # idx = j*1000+i
        kernel = gus[:,:,i]

    else :
        idx = i-4000
        kernel = gus[:,:,i]



    # else:
    #     idx = i
    #     kernel = gus[:, :, idx]

    # else:
    #     #kernel = gus[:, :, 5500]
    #     kernel = scipy.io.loadmat('kernel06.mat')
    #     kernel = np.array(kernel['f'])
    NoiseLevel = 0.001
    h,w = image_.shape
    input_=cv2.filter2D(image_,-1,kernel)+NoiseLevel*np.random.randn(h,w)
    #input_ = cv2.filter2D(image_, -1, kernel)


    input_train = input_
    label_train = label_



    return input_train,label_train
def preprocessd_test(path,config):

    """
    this preprocess :
    change to YCbCr
    blur image
    normalize
    """
##
    image_3 = Image.open(path)
    image_3 = image_3.convert("YCbCr")
    image,cb,cr = image_3.split()
    image = np.array(image,'f')
    cb = np.array(cb,'f')
    cr = np.array(cr,'f')

    image_= image / 255.
    label_ =image / 255.

    kernel=scipy.io.loadmat('kernel05.mat')
    kernel = np.array(kernel['f'])

    NoiseLevel = 0.001
    h, w = image_.shape
    input_ = cv2.filter2D(image_, -1, kernel) + NoiseLevel * np.random.randn(h, w)

    return input_, label_

def preprocessl(path,config):

    """
    this preprocess :
    change to YCbCr
    blur image
    normalize
    """
##
    image_3 = Image.open(path)
    image_3 = image_3.convert("YCbCr")
    image,cb,cr = image_3.split()
    image = np.array(image,'f')
    cb = np.array(cb,'f')
    cr = np.array(cr,'f')

    image_= image / 255.
    label_ =image / 255.

    kernel=scipy.io.loadmat('kernel05.mat')
    kernel = np.array(kernel['f'])

    NoiseLevel = 0.001
    h, w = image_.shape
    input_ = cv2.filter2D(image_, -1, kernel) + NoiseLevel * np.random.randn(h, w)
    blur = input_ * 255
    blur = Image.fromarray(np.uint8(blur))
    cb_ = Image.fromarray(np.uint8(cb))
    cr_ = Image.fromarray(np.uint8(cr))
    blur = Image.merge("YCbCr", (blur, cb_, cr_))
    blur = blur.convert("RGB")
    name = os.path.split(path)[-1]
    image_path = os.path.join(os.getcwd(), config.sample_dir)
    # image_path = os.path.join(image_path, "blur",name)
    image_path = os.path.join(image_path, "ls" + name)
    # # plt.imshow(blur)
    # # plt.show()
    blur.save(image_path)
    #input_ = scipy.ndimage.interpolation.zoom(label_, (1. / scale), prefilter=False)
    #input_ = scipy.ndimage.interpolation.zoom(input_, (scale / 1.), prefilter=False)


    return cb,cr,image,name
def prepare_data(sess, dataset):
    """
    Args:
      dataset: choose train dataset or test dataset

      For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
    """
    if FLAGS.is_train:
        filenames = os.listdir(dataset)
        data_dir = os.path.join(os.getcwd(), dataset)
        data = glob.glob(os.path.join(data_dir, "*.jpg"))
    else:
        #data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)), "Set5")
        data_dir = os.path.join(os.getcwd(), dataset)
        data = glob.glob(os.path.join(data_dir, "*.png"))

    return data


def make_data(sess, data, label):
    """
    Make input data as h5 file format
    Depending on 'is_train' (flag value), savepath would be changed.
    """
    if FLAGS.is_train:
        savepath = os.path.join(os.getcwd(), 'checkpoint/train.h5')
    else:
        savepath = os.path.join(os.getcwd(), 'checkpoint/test.h5')

    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('label', data=label)


def imread(path, is_grayscale=True):
    """
    Read image using its path.
    Default value is gray-scale, and image is read by YCbCr format as the paper said.
    :type is_grayscale: d
    """
    if is_grayscale:
        return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
    else:
        return scipy.misc.imread(path, mode='YCbCr').astype(np.float)





def train_input(data,config):
    """
    Read image files and make their sub-images and saved them as a h5 file format.
    """
    # Load data path
    # if config.is_train:
    #     data = prepare_data(sess, dataset="val2COCO")
    # else:
    #     data = prepare_data(sess, dataset="testC0C0")

    sub_input_sequence = []
    sub_label_sequence = []
    # sub_input_sequence_test = []
    # sub_label_sequence_test = []
    padding = abs(config.image_size - config.label_size) / 2  # 6

    if config.is_train:
        for j in range(0,1):
            for i in xrange(len(data)):
                input_train, label_train = preprocessd_train(data[i],i)
                print j,i
                if len(input_train.shape) == 3:
                    h, w, _ = input_train.shape
                else:
                    h, w = input_train.shape

                for x in range(0, h - config.image_size + 1, config.stride):
                    for y in range(0, w - config.image_size + 1, config.stride):
                        sub_input = input_train[x:x + config.image_size, y:y + config.image_size]  # [33 x 33]
                        sub_label = label_train[x + padding:x + padding + config.label_size,
                                    y + padding:y + padding + config.label_size]  # [21 x 21]
                        # sub_input_test = input_test[x:x + config.image_size, y:y + config.image_size]  # [33 x 33]
                        # sub_label_test = label_test[x + padding:x + padding + config.label_size,
                        #             y + padding:y + padding + config.label_size]
                        # Make channel value
                        sub_input = sub_input.reshape([config.image_size, config.image_size, 1])
                        sub_label = sub_label.reshape([config.label_size, config.label_size, 1])
                        # sub_input_test = sub_input_test.reshape([config.image_size, config.image_size, 1])
                        # sub_label_test = sub_label_test.reshape([config.label_size, config.label_size, 1])

                        sub_input_sequence.append(sub_input)
                        sub_label_sequence.append(sub_label)
                        # sub_input_sequence_test.append(sub_input)
                        # sub_label_sequence_test.append(sub_label)


    else:
         return
    return sub_input_sequence, sub_label_sequence

def test_input(data, config,n):
    sub_input_sequence = []
    sub_label_sequence = []
    padding = abs(config.image_size - config.label_size) / 2  # 6
    input_, label_ = preprocessd_test(data[n], config)

    if len(input_.shape) == 3:
        h, w, _ = input_.shape
    else:
        h, w = input_.shape

    # Numbers of sub-images in height and width of image are needed to compute merge operation.
    nx = ny = 0
    for x in range(0, h - config.image_size + 1, config.stride):
        nx += 1
        ny = 0
        for y in range(0, w - config.image_size + 1, config.stride):
            ny += 1
            sub_input = input_[x:x + config.image_size, y:y + config.image_size]  # [33 x 33]
            sub_label = label_[x + padding:x + padding + config.label_size,
                        y + padding:y + padding + config.label_size]  # [21 x 21]

            sub_input = sub_input.reshape([config.image_size, config.image_size, 1])
            sub_label = sub_label.reshape([config.label_size, config.label_size, 1])
            # if x ==0 and y==0:
            #     print sub_label
            sub_input_sequence.append(sub_input)
            sub_label_sequence.append(sub_label)
    return nx, ny, sub_input_sequence, sub_label_sequence



def merge(images, size,config):
    x = 0
    y = 0
    ny=size[1]
    h, w = images.shape[1], images.shape[2]
    pad = abs(config.image_size - config.label_size) / 2
    img = np.zeros([size[2]-2*pad, size[3]-2*pad] )
    # h,w = size[2],size[3]
    sub = config.label_size - config.stride
    #id = 0
    #img=np.array([[[0. for i in range(h0 * (size[0]+1))] for i in range(w0 * (size[1]+1))] for i in range(1)])
    #sub_images = {}
    for idx, image in enumerate(images):
        #sub_images.setdefault(idx,image)
        # y = idx % size[1]
        # x = idx // size[1]
        # if idx == 0:
        #     print image
        if x==0 and y==0 :
            img[x:x+config.label_size,y:y+config.label_size]=img[x:x+config.label_size,y:y+config.label_size]+image
        elif x==0 and y!=0:
            img[x:x + config.label_size,y:y + sub] = (img[x:x + config.label_size,y:y + sub] + image[0:config.label_size,0:sub])/2.
            img[x:x + config.label_size, y+sub:y + config.label_size] = \
                img[x:x + config.label_size,y+sub:y + config.label_size] + image[0:config.label_size,sub:config.label_size]

        elif x!=0 and y==0:
            img[x:x + sub, y:y + config.label_size] = (img[x:x + sub, y:y + config.label_size] + image[0:sub,0:config.label_size]) / 2.


            img[x+sub:x + config.label_size, y :y + config.label_size] = \
            img[x+sub:x + config.label_size, y :y + config.label_size] + image[sub:config.label_size,0:config.label_size]
        else:
            img[x:x + sub, y:y + config.label_size] = \
                (img[x:x + sub, y:y + config.label_size] + image[0:sub, 0:config.label_size]) / 2.
            img[x+sub:x + config.label_size, y:y + sub] = \
                (img[x+sub:x + config.label_size, y:y + sub] + image[sub:config.label_size, 0:sub]) / 2.
            img[x+sub:x+config.label_size,y+sub:y+config.label_size]=\
            img[x+sub:x+config.label_size,y+sub:y+config.label_size]+image[sub:config.label_size,sub:config.label_size]
    # for x in range(0, h - config.image_size + 1, config.stride):
    #
    #     for y in range(0, w - config.image_size + 1, config.stride):
    #         img[x:x+config.label_size,y:y+config.label_size]= img[x:x+config.label_size,y:y+config.label_size]+images[id,:,:]
    #         id +=1


        y+= config.stride
        if  (idx+1) % ny == 0:
            x += config.stride
            y = 0
    return img,pad




def lrelu(x, leak=0.3, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def next_batch(batch_size,images,labels,epoch):


    ep = 0

    while ep <= epoch:

        batch_idxs = len(images) // batch_size

        for idxs in xrange(0, batch_idxs):
            batch_images = images[idxs * batch_size:(idxs + 1) * batch_size]
            batch_labels = labels[idxs * batch_size:(idxs + 1) * batch_size]

            yield batch_images, batch_labels, idxs, ep
        ep += 1

    #print "Optimization done"


def compute_psnr(im,deblur):

    imdff = im - deblur
    rmse = np.sqrt(np.mean(np.square(imdff)))
    psnr = 20*math.log10(255/rmse)
    return psnr