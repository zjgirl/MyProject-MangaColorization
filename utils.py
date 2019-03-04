import tensorflow as tf
import numpy as np
import cv2
from glob import glob
from random import randint
import os

root_data = "./tfrecords/"
NUM_FOLDER = 28

class layer_norm(object):
    def __init__(self, name="layer_norm"):
        with tf.variable_scope(name): #生成一个上下文管理器，创建变量
            self.name = name

    def __call__(self, x, train=True): #__call__函数可将类实例转变成一个函数
        return tf.contrib.layers.layer_norm(x,scope=self.name)
   
class batch_norm(object):
    # h1 = lrelu(tf.contrib.layers.batch_norm(conv2d(h0, self.df_dim*2, name='d_h1_conv'),decay=0.9,updates_collections=None,epsilon=0.00001,scale=True,scope="d_h1_conv"))
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name): #生成一个上下文管理器，创建变量
            self.epsilon = epsilon #避免除零
            self.momentum = momentum #衰减系数
            self.name = name

    def __call__(self, x, train=True): #__call__函数可将类实例转变成一个函数
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, scope=self.name)
        
batchnorm_count = 0
def bnreset():
    global batchnorm_count
    batchnorm_count = 0
def bn(x):
    global batchnorm_count
    batch_object = batch_norm(name=("bn" + str(batchnorm_count))) #创建一个batch_norm对象
    batchnorm_count += 1 #用于命名
    return batch_object(x) #调用__call__函数

#卷积层定义方法
def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev)) #tf.get_variable()创建或获取变量，变量名一定要有
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

#反卷积过程
def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return deconv, w, biases
        else:
            return deconv

#leakyRelu激活函数的简单实现
def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

#全连接层
def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

 #返回标准大小的图像256*256
def get_image(image_path,size = 256):
    return transform(imread(image_path),size)

def transform(image,size, npx=512, is_crop=True):
    cropped_image = cv2.resize(image, (size,size))
    return np.array(cropped_image)

def imread(path):
    readimage = cv2.imread(path, 1)
    return readimage
#将batch中的图像进行拼接保存
def merge_color(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):#enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 1))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image

    return img[:,:,0]

#保存图像
def ims(name, img):
    print ("saving img " + name)
    cv2.imwrite(name, img*255)

def get_colorPic(isColored,pic_m):
    corlor_pic = 1 - np.zeros(np.shape(pic_m))
    corlor_pic[isColored] = pic_m[isColored]
    return corlor_pic


def colorGen(cimg, imagesize = 256, blocksize = 8):
    w, h, c = np.shape(cimg)
    hint = 255 - np.zeros((w, h, c))

    for i in range(30):
        randx = randint(0, imagesize - blocksize - 1)
        randy = randint(0, imagesize - blocksize - 1)
        hint[randx:randx + blocksize, randy:randy + blocksize] = np.mean(np.mean(cimg[randx:randx + blocksize, randy:randy + blocksize], axis=0),axis=0)
    return hint