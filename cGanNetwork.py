import tensorflow as tf
import numpy as np
import os
from glob import glob
import sys
import math
from random import randint
import utils
from utils import *
from utils import _tensor_size
from vgg19_net import VGG
from CreateSketchModel import *

imgs = "E:\\1myproject\\ColorNet\\code\\deepColor\\deepColor\\imgs_download\\imgs"
imgs_edge = "E:\\1myproject\\ColorNet\\code\\deepColor\\deepColor\\imgs_download\\imgs_edge"
train_results = "E:\\1myproject\\ColorNet\\1projectCode\\MyProject-MangaColorization\\results"
vgg_mat = "imagenet-vgg-verydeep-19.mat"

CONTENT_LAYER = 'relu3_3'
STYLE_LAYERS = ('relu1_2', 'relu2_2', 'relu3_3', 'relu4_3')

class Color():

    def __init__(self, model_tensor=(None,None), width=256, height=256, batchsize=16):

        self.images_tensor, self.feature_tensor = model_tensor

        self.batch_size = batchsize

        self.batch_size_sqrt = int(math.sqrt(self.batch_size))

        self.width = width

        self.height = height

        self.feature_height = int( int(int(height / 2 +0.5) / 2 + 0.5) / 2 +0.5)#特征size，在输入的第三层加入特征
        self.feature_width = int( int(int(width / 2 +0.5) / 2 + 0.5) / 2 +0.5)

        self.feature_dim = 256 #特征深度

        self.global_step = tf.Variable(0,trainable=False)

        self.gf_dim = 64

        self.df_dim = 64

        self.input_colors = 1

        self.input_colors2 = 3

        self.output_colors = 3

        self.l1_scaling = 100

        self.tv_scaling = 600

        self.vgg_scaling = 1

        self.d_bn1 = batch_norm(name='d_bn1')

        self.d_bn2 = batch_norm(name='d_bn2')

        self.d_bn3 = batch_norm(name='d_bn3')

        self.d_bn4 = batch_norm(name='d_bn4')

        self.line_images = tf.placeholder(tf.float32, [self.batch_size, self.height, self.width, self.input_colors])

        self.color_images = tf.placeholder(tf.float32, [self.batch_size, self.height, self.width, self.input_colors2])

        self.real_images = tf.placeholder(tf.float32, [self.batch_size, self.height, self.width, self.output_colors])

        #添加特征输入的placeholder，将在生成器的第3层加入特征
        self.line_features = tf.placeholder(tf.float32, [self.batch_size, self.feature_height, self.feature_width, self.feature_dim])

        #下面的拼接相当于cGAN中的条件，条件包括线稿图和笔触图

        #将线条图像、笔触图像在第4维进行拼接,此时第4位共有7个数
        combined_preimage = tf.concat((self.line_images, self.color_images),3)

        #将线图和笔触图拼接后的图像作为初始图像送入生成器
        self.generated_images = self.generator(combined_preimage)

        #计算vgg特征损失
        vgg = VGG(vgg_mat)
        self.real_vgg = vgg.net(vgg.preprocess(self.real_images))
        self.gen_vgg = vgg.net(vgg.preprocess(self.generated_images))
        self.vgg_loss = self.vgg_scaling * (2 * tf.nn.l2_loss(self.real_vgg[CONTENT_LAYER] - self.gen_vgg[CONTENT_LAYER]) / (_tensor_size(self.real_vgg[CONTENT_LAYER]))) / self.batch_size

        #拼接图像： 线条图像、笔触图像、真实图像，共有10个通道
        #拼接图像是为了能将线图传进去，作为条件c
        self.real_AB = tf.concat((combined_preimage, self.real_images),3) #真图

        #拼接图像：线条图像、模糊图像、生成器产生的图像
        self.fake_AB = tf.concat((combined_preimage, self.generated_images),3) #假图


        with tf.variable_scope("for_reuse_scope"):#很奇怪，这里不加这个调用优化函数时会报参数创建的错

            # 因为真实数据和生成数据都要经过判别器，所以需要指定 reuse 是否可用
            self.disc_true, disc_true_logits = self.discriminator(self.real_AB, reuse=False) #创建

            self.disc_fake, disc_fake_logits = self.discriminator(self.fake_AB, reuse=True)

        #disc_true_logits和disc_fake_logits是在最后没有加sigmoid函数的
        #对了，这里是在教判别器对真图像的判别概率为1，对假图像的判别概率为0
        #之前的分类任务有标签，放在一条语句中，这里需要将真图假图分开，分别给它们生成1/0标签，教它判断
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_true_logits,labels = tf.ones_like(disc_true_logits)))

        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_fake_logits,labels = tf.zeros_like(disc_fake_logits)))

        #self.d_loss_real = tf.reduce_mean(tf.scalar_mul(-1,disc_true_logits)) #根据公式推导
        #self.d_loss_fake = tf.reduce_mean(disc_fake_logits)
        self.d_loss = self.d_loss_real + self.d_loss_fake #判别器的Loss，最大化两类图像的判别分数(假图分数接近0，真图分数接近1)

        #生成器的Loss，最大化假图的判别分数（接近1），并且加入L1损失
        #这里是在教生成器，教它尽量生成能被判为1的假图像，如何生成？最小化真图和假图之间的L1损失
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_fake_logits,labels = tf.ones_like(disc_fake_logits)))
        self.l1_loss =  self.l1_scaling * tf.reduce_mean(tf.abs(self.real_images - self.generated_images))

        # 全局方差损失
        tv_y_size = self._tensor_size(self.generated_images[:, 1:, :, :])
        tv_x_size = self._tensor_size(self.generated_images[:, :, 1:, :])
        y_tv = tf.nn.l2_loss(self.generated_images[:, 1:, :, :] - self.generated_images[:, :self.height - 1, :, :])
        x_tv = tf.nn.l2_loss(self.generated_images[:, :, 1:, :] - self.generated_images[:, :, :self.width - 1, :])
        self.tv_loss = self.tv_scaling * (x_tv / tv_x_size + y_tv / tv_y_size) / self.batch_size

        self.g_loss += (self.tv_loss + self.l1_loss + self.vgg_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name] #这里变量名起到很大作用，区分生成器的变量和判别器的变量，分开训练

        self.g_vars = [var for var in t_vars if 'g_' in var.name]


        self.d_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars,global_step = self.global_step)

        self.g_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)

        tf.summary.scalar('d_loss', self.d_loss)
        tf.summary.scalar('g_loss', self.g_loss)
        #self.clip_d_op = [var.assign(tf.clip_by_value(var,CLIP[0],CLIP[1])) for var in self.d_vars]

    def _tensor_size(self, tensor):
        from operator import mul
        import functools
        return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)

    #判别器,卷积层抽取特征，最后加一个全连接层进行分类
    def discriminator(self, image, y=None, reuse=False):
        # image is 256 x 256 x (input_c_dim + output_c_dim)
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        #定义conv2d卷积方法时已经定义了是全零填充，步长为2，所以这里输出维度减半
        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv')) # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv'))) # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv'))) # h2 is (32 x 32 x self.df_dim*4)

        hn1 = tf.concat((h2,self.line_features),3) #add feature: en is (32 x 32 x 512)
        hn2 = lrelu(self.d_bn3(conv2d(hn1,self.df_dim*4,d_h=1,d_w=1,name='d_hn2_conv'))) #en2 is (32 x 32 x256)

        h3 = lrelu(self.d_bn4(conv2d(hn2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv'))) # h3 is (16 x 16 x self.df_dim*8)
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin') #加一层全连接层，输出为一个列向量，即batch中每个图像的判别概率
        #return tf.nn.sigmoid(h4), h4
        return None,h4

    #生成器，输入是线图（+模糊图）
    #先下采样再上采样，U-Net网络结构
    def generator(self, img_in):
        sw = self.width ; sh = self.height
        sw2 = int(sw/2+0.5); sw4 = int(sw2/2+0.5);sw8 = int(sw4/2+0.5); sw16 = int(sw8/2+0.5)
        sh2 = int(sh / 2 + 0.5); sh4 = int(sh2 / 2 + 0.5); sh8 = int(sh4 / 2 + 0.5); sh16 = int(sh8 / 2 + 0.5)

        # image is (256 x 256 x input_c_dim)
        e1 = conv2d(img_in, self.gf_dim, name='g_e1_conv') # e1 is (128 x 128 x 64)
        e2 = bn(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv')) # e2 is (64 x 64 x 128)
        e3 = bn(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv')) # e3 is (32 x 32 x 256)
        
        #还有多种拼接方法可以尝试
        en1 = tf.concat((e3,self.line_features),3) #add feature: en is (32 x 32 x 512)
        en2 = bn(conv2d(lrelu(en1),self.gf_dim*4,d_h=1,d_w=1,name='g_en2_conv')) #en2 is (32 x 32 x256)

        e4 = bn(conv2d(lrelu(en2), self.gf_dim*8, name='g_e4_conv')) # e4 is (16 x 16 x 512)
        e5 = bn(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv')) # e5 is (8 x 8 x 512)


        #self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(e5), [self.batch_size, sh16, sw16, self.gf_dim*8], name='g_d4', with_w=True) #16*16*512
        # 1. deconv -> resize + conv
        self.d4 = tf.image.resize_images(e5,[sh16, sw16],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) #(16 x 16 x 512)
        self.d4 = conv2d(lrelu(self.d4), self.gf_dim * 8, d_h=1, d_w=1, name='g_d4')  # （16 x 16 x 512 ）
        d4 = bn(self.d4) #bn是对每一层的参数进行标准化

        #add skip connections
        d4 = tf.concat((d4, e4),3) #将还原的图像与特征图像进行拼接，U型结构的特点，为了精准定位
        # d4 is (16 x 16 x self.gf_dim*8*2)

        #self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4), [self.batch_size, sh8, sw8, self.gf_dim*4], name='g_d5', with_w=True)
        # 2. deconv -> resize + conv
        self.d5 = tf.image.resize_images(d4, [sh8, sw8], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # (32 x 32 x 1024)
        self.d5 = conv2d(lrelu(self.d5), self.gf_dim * 4, d_h=1, d_w=1, name='g_d5') # （32 x 32 x 256 ）

        d5 = bn(self.d5)
        d5 = tf.concat((d5, en2),3)
        # d5 is (32 x 32 x self.gf_dim*4*2)

        self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5), [self.batch_size, sh4, sw4, self.gf_dim*2], name='g_d6', with_w=True)
        d6 = bn(self.d6)
        d6 = tf.concat((d6, e2), 3)
        # d6 is (64 x 64 x self.gf_dim*2*2)

        self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6), [self.batch_size, sh2, sw2, self.gf_dim], name='g_d7', with_w=True)
        d7 = bn(self.d7)
        d7 = tf.concat((d7, e1), 3)
        # d7 is (128 x 128 x self.gf_dim*1*2)

        self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7), [self.batch_size, sh, sw, self.output_colors], name='g_d8', with_w=True) #最后一层输出为3通道图像
        # d8 is (256 x 256 x output_c_dim) 最后的生成图像是3个通道的

        return tf.nn.tanh(self.d8) #最后一层的激活函数为tanh函数


    def train(self):

        self.loadmodel()

        data = []
        edge = []
        #for i in range(1, 29):
        d = glob(os.path.join(imgs, str(1), "*.jpg"))
        data.extend(d)
        e = glob(os.path.join(imgs_edge, str(1), "*.jpg"))
        edge.extend(e)

        base = np.array([get_image(sample_file) for sample_file in data[0:self.batch_size]])  # 返回标准大小的图像batch
        line = np.array([get_image(sample_file) for sample_file in edge[0:self.batch_size]])

        base_normalized = base / 255.0
        line_normalized = line / 255.0

        # 获取线图,线图只有一个通道，所以后面要扩维
        base_edge = line_normalized[:, :, :, 0]
        base_edge = np.expand_dims(base_edge, 3)

        # 笔触生成
        base_colors = np.array([colorGen(ba) for ba in base]) / 255.0

        # 这里保存的是将batch中的图像拼接到一起，如batch=4，则上2张，下2张
        ims(train_results + "/base.png", merge_color(base_normalized, [self.batch_size_sqrt, self.batch_size_sqrt]))

        ims(train_results + "/base_line.jpg", merge(base_edge, [self.batch_size_sqrt, self.batch_size_sqrt]))

        ims(train_results + "/base_colors.jpg", merge_color(base_colors, [self.batch_size_sqrt, self.batch_size_sqrt]))

        datalen = len(data)  # 获取训练图像总数

        for e in range(20):  # 将所有训练数据循环几遍

            for i in range(datalen // self.batch_size):  # 这里循环获取所有训练图像组成batch，每次循环按序获取一组新的batch

                k = randint(0, datalen // self.batch_size - 1)
                batch_files = data[k * self.batch_size:(k + 1) * self.batch_size]  # 获取当前batch图像

                batch = np.array([get_image(batch_file) for batch_file in batch_files])  # 获取图像值并标准化为256*256
                # 彩色真图
                batch_normalized = batch / 255.0

                # 线稿图
                edge_files = edge[k * self.batch_size:(k + 1) * self.batch_size]  # 获取当前batch图像

                batch_edge = np.array([get_image(edge_file) for edge_file in edge_files]) / 255.0

                batch_edge = np.expand_dims(batch_edge[:, :, :, 0], 3)

                # 笔触图
                batch_colors = np.array([colorGen(ba) for ba in batch]) / 255.0
				
                # 特征图,需要计算扩展的维度
                batch_features = self.sess.run(self.feature_tensor,feed_dict={self.images_tensor:batch_edge})
                feashape = np.shape(batch_features)
                expandHeight = int((feashape[1]-(self.feature_height - feashape[1]))//2) #想要拼接中间的部分，获得应该从第几行开始抽取
                expandWidth = int((feashape[2]-(self.feature_width - feashape[2]))//2)
                concate_1 = np.concatenate((batch_features, batch_features[:, expandHeight:(feashape[1] - expandHeight), :, :]), 1)
                batch_features = np.concatenate((concate_1, concate_1[:, :, expandWidth:(feashape[2] - expandWidth), :]), 2)

                # feed进去的依次是原彩色图、线图、模糊图，这里是在分别训练判别器和生成器

                d_loss, _, step = self.sess.run([self.d_loss, self.d_optim, self.global_step],
                                                feed_dict={self.real_images: batch_normalized,
                                                           self.line_images: batch_edge,
                                                           self.color_images: batch_colors,
                                                           self.line_features: batch_features})
                # self.sess.run(self.clip_d_op)

                #if i % 2 == 0:
                g_loss, _, l1_loss, tv_loss, vgg_loss = self.sess.run([self.g_loss, self.g_optim, self.l1_loss, self.tv_loss, self.vgg_loss],
                                                          feed_dict={self.real_images: batch_normalized,
                                                                     self.line_images: batch_edge,
                                                                     self.color_images: batch_colors,
                                                                     self.line_features: batch_features})

                print("%d: [%d / %d] d_loss %f, g_loss %f, l1_loss %f, tv_loss %f, vgg_loss %f" % (e, i, (datalen / self.batch_size), d_loss, g_loss, l1_loss, tv_loss, vgg_loss))

                # 每50次打印一下生成器的生成图片结果
                if i % 50 == 0:
                    recreation = self.sess.run(self.generated_images, feed_dict={self.real_images: base_normalized,
                                                                                 self.line_images: base_edge,
                                                                                 self.color_images: base_colors,
                                                                                 self.line_features: batch_features})

                    ims(train_results + '/' + str(step) + ".jpg",
                        merge_color(recreation, [self.batch_size_sqrt, self.batch_size_sqrt]))

                    # self.save("./checkpoint", step)
                    result = self.sess.run(self.merged,
                                           feed_dict={self.real_images: batch_normalized, 
													  self.line_images: batch_edge,
                                                      self.color_images: batch_colors,
													  self.line_features: batch_features})  # 计算需要写入的日志数据
                    #self.writer.add_summary(result, step)  # 将日志数据写入文件
                if (e != 0 and i == 0) or (e == 9 and i == datalen // self.batch_size - 1):
                    self.save("./checkpoint", step)

    def loadmodel(self, load_discrim=True):

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

        self.merged = tf.summary.merge_all()  # 将图形、训练过程等数据合并在一起
        #self.writer = tf.summary.FileWriter('logs', self.sess.graph)  # 将训练日志写入到logs文件夹下

        if load_discrim:

            self.saver = tf.train.Saver()

        else:

            self.saver = tf.train.Saver(self.g_vars)



        if self.load("./checkpoint"):

            print("Loaded")

        else:

            print ("Load failed")



    def sample(self):

        self.loadmodel(False)

        data = glob(os.path.join("../test_data", "*_o.*"))

        data_m = glob(os.path.join("../test_data", "*_m.*"))

        datalen = len(data)

        for i in range(min(100,datalen // self.batch_size)):

            batch_files = data[i]
            batch_files_m = data_m[i]

            batch_file = imread(batch_files)
            batch_file_m = imread(batch_files_m)
            h, w, c = np.shape(batch_file) #原来的宽高

            batch = np.array([cv2.resize(batch_file, (self.height,self.width))])
            batch_m = np.array([cv2.resize(batch_file_m, (self.height,self.width))])

            batch_normalized = batch/255.0 #实际上生成器不会用到这个

            batch_edge = batch_normalized[:,:,:,0]

            batch_edge = np.expand_dims(batch_edge, 3)

            isColored = abs(batch_m / 255.0 - batch_normalized) > 0.1
            batch_colors = get_colorPic(isColored, batch_m / 255.0)

            # 特征图,需要计算扩展的维度
            batch_features = self.sess.run(self.feature_tensor, feed_dict={self.images_tensor: batch_edge})
            feashape = np.shape(batch_features)
            expandHeight = int((feashape[1] - (self.feature_height - feashape[1])) // 2)  # 想要拼接中间的部分，获得应该从第几行开始抽取
            expandWidth = int((feashape[2] - (self.feature_width - feashape[2])) // 2)
            concate_1 = np.concatenate((batch_features, batch_features[:, expandHeight:(feashape[1] - expandHeight), :, :]), 1)
            batch_features = np.concatenate((concate_1, concate_1[:, :, expandWidth:(feashape[2] - expandWidth), :]), 2)

            #batch_edge = batch_normalized[:,:,:,0]
            #batch_edge = np.expand_dims(batch_edge, 3)

            ims("results/colors_"+str(i)+".jpg",merge_color(batch_colors, [self.batch_size_sqrt, self.batch_size_sqrt]))

            recreation = self.sess.run(self.generated_images, feed_dict={self.real_images: batch_normalized,
                                                                         self.line_images: batch_edge,
                                                                         self.color_images: batch_colors,
                                                                         self.line_features: batch_features})

            #for shadding
            shading = np.tile(np.expand_dims(np.expand_dims(gausBlur(batch_edge[0] * 255), axis=0),axis=3),(1, 1, 1, 3))
            recreation = (recreation * 255 - (255 - shading) / 3) / 255.0

            recreation = np.expand_dims(cv2.resize(recreation[0],(w, h)),0) #恢复到原来的分辨率
            ims("results/sample_"+str(i)+".jpg",merge_color(recreation, [self.batch_size_sqrt, self.batch_size_sqrt]))

            ims("results/edge_"+str(i)+"_line.jpg",merge_color(batch_edge, [self.batch_size_sqrt, self.batch_size_sqrt]))


    def save(self, checkpoint_dir, step):

        model_name = ""

        model_dir = "checkboard"

        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):

            os.makedirs(checkpoint_dir)



        self.saver.save(self.sess,

                        os.path.join(checkpoint_dir, model_name),

                        global_step=step)



    def load(self, checkpoint_dir):

        print(" [*] Reading checkpoint...")

        model_dir = "checkboard"

        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)


        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:

            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))

            return True

        else:

            return False


