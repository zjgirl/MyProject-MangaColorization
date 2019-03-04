import tensorflow as tf
from cGanNetwork import Color
from CreateSketchModel import *

if __name__ == '__main__':

    print("please input train or sample?")

    cmd = input()

    if cmd == "train":
        createModel(batchSize=16, imageSize=256)
        images_tensor, feature_tensor = readModel()
        c = Color(model_tensor=(images_tensor, feature_tensor))
        c.train()

    elif cmd == "sample":
        createModel(batchSize=1, imageSize=512)
        images_tensor, feature_tensor = readModel()
        c = Color(model_tensor=(images_tensor, feature_tensor), imgsize=512, batchsize=1)
        c.sample()

    else:

        print ("Usage: python main.py [train, sample]")