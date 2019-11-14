import tensorflow as tf
from cGanNetwork import Color
from CreateSketchModel import *
from glob import glob
import os
import numpy as np
from utils import *

if __name__ == '__main__':

    print("please input train / sample ?")

    cmd = input()

    if cmd == "train":
        createModel(batchSize=16, width=256, height=256)
        images_tensor, feature_tensor = readModel()
        c = Color(model_tensor=(images_tensor, feature_tensor))
        c.train()

    elif cmd == "sample":
        wh = 512
        createModel(batchSize=1, width=wh, height=wh)
        images_tensor, feature_tensor = readModel()
        c = Color(model_tensor=(images_tensor, feature_tensor),width=wh, height=wh,batchsize=1)
        c.sample()
    '''
    elif cmd == "sample":

        data = glob(os.path.join("../test_data", "125_o.*"))

        data_m = glob(os.path.join("../test_data", "125_m.*"))

        datalen = len(data)

        for i in range(min(100, datalen)):

            batch_files = data[i : i + 1]
            batch_files_m = data_m[i : i + 1]

            # batch = np.array([cv2.resize(imread(batch_file), (self.image_size,self.image_size)) for batch_file in batch_files])
            # batch_m = np.array([cv2.resize(imread(batch_file_m), (self.image_size,self.image_size)) for batch_file_m in batch_files_m])
            batch = np.array([imread(batch_file) for batch_file in batch_files])
            batch_m = np.array([imread(batch_file_m) for batch_file_m in batch_files_m])

            shape = np.shape(batch)
            createModel(batchSize=1, width=shape[2], height=shape[1])
            images_tensor, feature_tensor = readModel()
            c = Color(model_tensor=(images_tensor, feature_tensor),width=shape[2], height=shape[1],batchsize=1)
            c.loadmodel(False)

            batch_normalized = batch / 255.0  # 实际上生成器不会用到这个

            batch_edge = batch_normalized[:, :, :, 0]

            batch_edge = np.expand_dims(batch_edge, 3)

            isColored = abs(batch_m / 255.0 - batch_normalized) > 0.1
            batch_colors = get_colorPic(isColored, batch_m / 255.0)

            # 特征图,需要计算扩展的维度
            batch_features = c.sess.run(feature_tensor, feed_dict={images_tensor: batch_edge})
            feashape = np.shape(batch_features)
            expandHeight = int((feashape[1] - (c.feature_height - feashape[1])) // 2)  # 想要拼接中间的部分，获得应该从第几行开始抽取
            expandWidth = int((feashape[2] - (c.feature_width - feashape[2])) // 2)
            concate_1 = np.concatenate((batch_features, batch_features[:, expandHeight:(c.feature_height - feashape[1] + expandHeight), :, :]), 1)
            batch_features = np.concatenate((concate_1, concate_1[:, :, expandWidth:(c.feature_width - feashape[2] + expandWidth), :]), 2)

            # batch_edge = batch_normalized[:,:,:,0]
            # batch_edge = np.expand_dims(batch_edge, 3)

            ims("results/colors_" + str(i) + ".jpg",
                merge_color(batch_colors, [c.batch_size_sqrt, c.batch_size_sqrt]))

            recreation = c.sess.run(c.generated_images, feed_dict={c.real_images: batch_normalized,
                                                                         c.line_images: batch_edge,
                                                                         c.color_images: batch_colors,
                                                                         c.line_features: batch_features})

            # for shadding
            shading = np.tile(np.expand_dims(np.expand_dims(gausBlur(batch_edge[0] * 255), axis=0), axis=3),(1, 1, 1, 3))
            recreation = (recreation * 255 - (255 - shading) / 3) / 255.0

            ims("results/sample_" + str(i) + ".jpg",merge_color(recreation, [c.batch_size_sqrt, c.batch_size_sqrt]))

            ims("results/edge_" + str(i) + "_line.jpg",merge_color(batch_edge, [c.batch_size_sqrt, c.batch_size_sqrt]))

            c.sess.close()
            tf.reset_default_graph() # 清除图结点，防止下次使用时说该结点已经存在
    else:

        print ("Usage: python main.py [train, sample]")
    '''