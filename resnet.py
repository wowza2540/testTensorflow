# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/resnet_in_tensorflow
# ==============================================================================
'''
This is the resnet structure
'''
import numpy as np
from hyper_parameters import *


BN_EPSILON = 0.001

def activation_summary(x):
    '''
    :param x: A Tensor
    :return: Add histogram summary and scalar summary of the sparsity of the tensor
    '''
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x) #tf.summary.histogram(name, values, collections=None)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
    #tf.nn.zero_fraction(x) = Returns the fraction of zeros in value.If value is empty, the result is nan.
    #name = optional , value = A tensor of numeric type.


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    #tf.contrib.layers.xavier_initializer() = Returns an initializer performing "Xavier" initialization for weights.
    #ปัญหาความยากในการ training deep 
    '''
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''
    
    ## TODO: to allow different weight decay to fully connected layer and conv layer
    regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay) 
    #Returns a function that can be used to apply L2 regularization to weights.ช่วยในการ training data.

    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    #Gets an existing variable with these parameters or create a new one.
    #shape: Shape of the new or existing variable., initializer = ตัวเริ่มต้น , redularizer = tensor/none
    return new_variables


def output_layer(input_layer, num_labels):
    '''
    :param input_layer: 2D tensor
    :param num_labels: int. How many output labels in total? (10 for cifar10 and 100 for cifar100)
    :return: output layer Y = WX + B
    '''
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,
                            initializer=(factor=1.0))
    #tf.uniform_unit_scaling_initializer เจนเนอเรทtensor  ทำหน้าที่เก็บค่า scale ของ input ใช้ input * W ซึ่ง W ได้มาจากการสุ่ม
    #dim = size of output คำนวณเหมือน convolution network ให้ผลที่คล้ายกัน
    #factor: Float. A multiplicative factor by which the values will be scaled.
    fc_b = create_variables(name='ftf.uniform_unit_scaling_initializerc_bias', shape=[num_labels], initializer=tf.zeros_initializer())
    #tf.zeros_initializer()=nitializer that generates tensors initialized to 0.

    fc_h = tf.matmul(input_layer, fc_w) + fc_b #Multiplies matrix a by matrix b, producing a * b.
    return fc_h


def batch_normalization_layer(input_layer, dimension):
    '''สรุปแล้วว tensor เป็นชนิดของตัวแปรใช่ป่าวหว่า
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: the 4D tensor after being normalized
    '''
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    #Calculate the mean and variance of input_layer. axes=Array of ints, compute mean and variance.
    beta = tf.get_variable('beta', dimension, tf.float32,initializer=tf.constant_initializer(0.0, tf.float32))
    #shape=dimension, dtype=tf.float32(The data type. Only floating point types are supported)
    #tf.constant_initializer(value,dtype,verify_shape=booleanเอาไว้พิสูจน์ shape of value)
    gamma = tf.get_variable('gamma', dimension, tf.float32,initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)
    #เอาไปใส่สูตรการคำนวณ (gamma(input_layer-mean))sigma + beta , sigma = BN_epsilon
    return bn_layer


def conv_bn_relu_layer(input_layer, filter_shape, stride):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number] มิติของ output
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''

    out_channel = filter_shape[-1] #<?>
    filter = create_variables(name='conv', shape=filter_shape)

    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
       '''
       create convolution level สร้างเป็น metrix ของ output
       filter : (int) มิติของ output
       strides : ก้าวในการขยับ(batch, width, height, channels)
       padding : valid / same
    '''
    bn_layer = batch_normalization_layer(conv_layer, out_channel)
   '''
       each input channel ต้องเปลี่ยนเป็น mini-batch 
       โดยช่องแรกจะลดจาก mini-batch เดิม ในเลเวลถัดไปใช้ offset(beta),scale(gamma) มาคำนวณ
       Use batch normalization layers between convolutional layers and nonlinearities
    '''
    output = tf.nn.relu(bn_layer) #bn_layer=A tensor
    return output


def bn_relu_conv_layer(input_layer, filter_shape, stride):
    '''
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
    แปลงจาก relu to conv
    '''

    in_channel = input_layer.get_shape().as_list()[-1]

    bn_layer = batch_normalization_layer(input_layer, in_channel)
    relu_layer = tf.nn.relu(bn_layer)

    filter = create_variables(name='conv', shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    return conv_layer



def residual_block(input_layer, output_channel, first_block=False):
    '''
    Defines a residual block in ResNet
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    '''
    input_channel = input_layer.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv1_in_block'):#value ใช้พิสูจน์ว่ากราปนี้มาจากกราฟเริ่มต้นจริงๆ 
        if first_block:
            filter = create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
            conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
        else:
            conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride)

    with tf.variable_scope('conv2_in_block'):
        conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1)

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                     input_channel // 2]])
    else:
        padded_input = input_layer

    output = conv2 + padded_input
    return output

def inference(input_tensor_batch, n, reuse):
    '''
    The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
    :param input_tensor_batch: 4D tensor
    :param n: num_residual_blocks
    :param reuse: To build train graph, reuse=False. To build validation graph and share weights
    with train graph, resue=True
    :return: last layer in the network. Not softmax-ed
    '''

    layers = []
    with tf.variable_scope('conv0', reuse=reuse):
        conv0 = conv_bn_relu_layer(input_tensor_batch, [3, 3, 3, 16], 1)
        activation_summary(conv0)
        layers.append(conv0)

    for i in range(n):
        with tf.variable_scope('conv1_%d' %i, reuse=reuse):
            if i == 0:
                conv1 = residual_block(layers[-1], 16, first_block=True)
            else:
                conv1 = residual_block(layers[-1], 16)
            activation_summary(conv1)
            layers.append(conv1)

    for i in range(n):
        with tf.variable_scope('conv2_%d' %i, reuse=reuse):
            conv2 = residual_block(layers[-1], 32)
            activation_summary(conv2)
            layers.append(conv2)

    for i in range(n):
        with tf.variable_scope('conv3_%d' %i, reuse=reuse):
            conv3 = residual_block(layers[-1], 64)
            layers.append(conv3)
        assert conv3.get_shape().as_list()[1:] == [8, 8, 64]

    with tf.variable_scope('fc', reuse=reuse):
        in_channel = layers[-1].get_shape().as_list()[-1]
        bn_layer = batch_normalization_layer(layers[-1], in_channel)
        relu_layer = tf.nn.relu(bn_layer)
        global_pool = tf.reduce_mean(relu_layer, [1, 2])

        assert global_pool.get_shape().as_list()[-1:] == [64]
        output = output_layer(global_pool, 10)
        layers.append(output)

    return layers[-1]


def test_graph(train_dir='logs'):
    '''
    Run this function to look at the graph structure on tensorboard. A fast way!
    :param train_dir:
    '''
    input_tensor = tf.constant(np.ones([128, 32, 32, 3]), dtype=tf.float32)
    #สร้างค่าคงที่ โดยใช้ dtype,value,shape np.ones=Creates a tensor
    result = inference(input_tensor, 2, reuse=False)
    init = tf.initialize_all_variables()
    #tf.initialize_variables อันบนเลิกใช้นานแล้ว ใช้อันนี้แทนAn Op that initializes global variables in the graph.
    sess = tf.Session()
    sess.run(init)
    summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)
