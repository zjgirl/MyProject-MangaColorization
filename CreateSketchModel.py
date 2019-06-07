import tensorflow as tf
from tensorflow.python.framework import graph_util
from migrate_SketchANet import inference
from migrate_dataLayer import load_pretrained_model
from tensorflow.python.platform import gfile

pb_file_path = "SketchANetModel/SketchModel.pb"
mat_path = 'SketchANetModel/model_with_order_info_256.mat'

def createModel(batchSize, width, height):
    weights, biases = load_pretrained_model(mat_path)
    images = tf.placeholder(tf.float32, [batchSize,height,width,1],name="images")

    _, feature = inference(images, dropout_prob=1.0, pretrained=(weights, biases), visualize=False)
    '''
    # 这里由于是从15变成32，直接2倍还差2行，所以用了最后一行进行再复制，不知道是否会加重其权重
    # 或者采用随机行？？
    # 注释掉是因为这部分工作改在训练过程中做
    expand_dim1 = tf.expand_dims(feature[:,14,:,:],1)
    feature_32_15 = tf.concat((feature, expand_dim1, feature, expand_dim1), 1)
    expand_dim2 = tf.expand_dims(feature_32_15[:,:,14,:],2)
    feature_32_32 = tf.concat((feature_32_15, expand_dim2, feature_32_15,expand_dim2), 2,name = "feature")
    '''
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        graph_def = tf.get_default_graph().as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(sess,graph_def,['images','L5/relu5'])
        with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
            f.write(output_graph_def.SerializeToString())

def readModel():
    with gfile.FastGFile(pb_file_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # 将graph_def中保存的图加载到当前图中，其中保存的时候保存的是计算节点的名称，为add
    # 但是读取时使用的是张量的名称所以是add:0
    images_tensor, feature_tensor = tf.import_graph_def(graph_def, return_elements=["images:0", "L5/relu5:0"])
    return images_tensor,feature_tensor