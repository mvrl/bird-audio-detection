# MAYBE explicitly name layers

import tensorflow as tf
slim = tf.contrib.slim
import numpy as np

def network_arg_scope(
        weight_decay=0.00004,
        is_training=True,
        batch_norm_var_collection='moving_vars',
        activation_fn=tf.nn.relu):
    ''' Sets default parameters for network layers.'''

    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.999,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # collection containing update_ops.
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        # collection containing the moving mean and moving variance.
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection],
        }
    }

    # for activation functions that are not "mostly linear" we should
    # have a scale parameter
    if activation_fn.func_name in ['elu']: 
        batch_norm_params['scale'] = True

    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope([slim.conv2d],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        weights_initializer=slim.variance_scaling_initializer(),
        padding='VALID',
        outputs_collections=tf.GraphKeys.ACTIVATIONS,
        activation_fn=activation_fn,
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.max_pool2d], 
            stride=(2,1),
            outputs_collections=tf.GraphKeys.ACTIVATIONS):
            with slim.arg_scope([slim.batch_norm], is_training=is_training) as sc:
                return sc

def network(net, is_training=True, activation_fn=tf.nn.relu, capacity=1.0, capacity2=1.0, network='v2.1'):

    print('Using network ' + network + '.')
    return networks[network](net, 
        is_training=is_training, 
        activation_fn=activation_fn, 
        capacity=capacity)

def network_v4(net, is_training=True, activation_fn=tf.nn.relu, capacity=1.0):

    Nb = net.get_shape()[0]

    net = tf.reshape(net,(-1,2000,200))
    net = tf.expand_dims(net,-1)

    with slim.arg_scope(network_arg_scope(is_training=is_training,
        activation_fn=activation_fn)):

        # extract window features
        net = slim.conv2d(net,np.rint(capacity*16),[1,16],stride=(1,4))
        net = slim.conv2d(net,np.rint(capacity*32),[1,16],stride=(1,2))
        net = slim.conv2d(net,np.rint(capacity*64),[1,16],stride=(1,2))
        net = tf.reduce_max(net,[2],keep_dims=True)

        # combine window features
        net = slim.conv2d(net,np.rint(capacity*128),[5,1],stride=(1,1))
        net = slim.conv2d(net,np.rint(capacity*256),[5,1],stride=(1,1))
        net = slim.conv2d(net,np.rint(capacity*256),[1,1],stride=(1,1))
        net = slim.conv2d(net,2,[8,1],normalizer_fn=None,activation_fn=None)

        net = slim.flatten(tf.reduce_max(net,[1]))

        net = tf.squeeze(net)

        return net 

def network_v3(net, is_training=True, activation_fn=tf.nn.relu, capacity=1.0):

    Nb = net.get_shape()[0]

    net = tf.reshape(net,(-1,2000,200))
    net = tf.expand_dims(net,-1)

    with slim.arg_scope(network_arg_scope(is_training=is_training,
        activation_fn=activation_fn)):

        # extract window features
        net = slim.conv2d(net,np.rint(capacity*100),[1,150],stride=(1,11))
        net = slim.conv2d(net,np.rint(capacity*100),[1,1],stride=(1,1))
        net = slim.conv2d(net,np.rint(capacity*100),[1,1],stride=(1,1))
        net = tf.reduce_max(net,[2],keep_dims=True)

        # combine window features
        net = slim.conv2d(net,np.rint(capacity*64),[5,1],stride=(1,1))
        net = slim.conv2d(net,np.rint(capacity*128),[5,1],stride=(1,1))
        net = slim.conv2d(net,2,[8,1],normalizer_fn=None,activation_fn=None)

        net = slim.flatten(tf.reduce_max(net,[1]))

        net = tf.squeeze(net)

        return net 

def network_v2_3(net, is_training=True, activation_fn=tf.nn.relu,
        capacity=1.0, capacity2=1.0):

    Nb = net.get_shape()[0]

    net = tf.reshape(net,(-1,2000,200))
    net = tf.expand_dims(net,-1)
    # add a log
    net = tf.concat(3,(net,tf.log(net+1+1e-6)))

    with slim.arg_scope(network_arg_scope(is_training=is_training,
        activation_fn=activation_fn)):

        # extract window features
        net = slim.conv2d(net,np.rint(capacity*8),[1,5],stride=(1,2))
        net = slim.conv2d(net,np.rint(capacity*16),[1,5],stride=(1,2))
        net = slim.conv2d(net,np.rint(capacity*32),[1,3],stride=(1,1))

        net = tf.reduce_max(net,[2],keep_dims=True)

        # combine window features
        net = slim.conv2d(net,np.rint(capacity*32),[9,1],stride=(2,1))
        net = slim.conv2d(net,np.rint(capacity*64),[9,1],stride=(2,1))
        net = slim.conv2d(net,np.rint(capacity*128),[9,1],stride=(2,1))
        net = slim.conv2d(net,2,[8,1],normalizer_fn=None,activation_fn=None)

        net = slim.flatten(tf.reduce_max(net,[1]))

        net = tf.squeeze(net)

        return net 

def network_v2_2(net, is_training=True, activation_fn=tf.nn.relu, capacity=1.0):

    Nb = net.get_shape()[0]

    net = tf.reshape(net,(-1,2000,200))
    net = tf.expand_dims(net,-1)
    # add a log
    net = tf.concat(3,(net,tf.log(net+1+1e-6)))

    with slim.arg_scope(network_arg_scope(is_training=is_training,
        activation_fn=activation_fn)):

        # extract window features
        net = slim.conv2d(net,np.rint(capacity*16),[1,21],stride=(1,2))
        net = slim.conv2d(net,np.rint(capacity*16),[1,11],stride=(1,2))
        net = slim.conv2d(net,np.rint(capacity*32),[1,5],stride=(1,2))
        net = tf.reduce_max(net,[2],keep_dims=True)

        # combine window features
        net = slim.conv2d(net,np.rint(capacity*64),[5,1],stride=(1,1))
        net = slim.conv2d(net,np.rint(capacity*128),[5,1],stride=(1,1))
        net = slim.conv2d(net,2,[8,1],normalizer_fn=None,activation_fn=None)

        net = slim.flatten(tf.reduce_max(net,[1]))

        net = tf.squeeze(net)

        return net 

def network_v2_1(net, is_training=True, activation_fn=tf.nn.relu, capacity=1.0):

    Nb = net.get_shape()[0]

    net = tf.reshape(net,(-1,2000,200))
    net = tf.expand_dims(net,-1)
    # add a log
    net = tf.concat(3,(net,tf.log(net+1+1e-6)))

    with slim.arg_scope(network_arg_scope(is_training=is_training,
        activation_fn=activation_fn)):

        # extract window features
        net = slim.conv2d(net,np.rint(capacity*8),[1,5],stride=(1,2))
        net = slim.conv2d(net,np.rint(capacity*16),[1,5],stride=(1,2))
        net = slim.conv2d(net,np.rint(capacity*32),[1,5],stride=(1,2))
        net = tf.reduce_max(net,[2],keep_dims=True)

        # combine window features
        net = slim.conv2d(net,np.rint(capacity*64),[5,1],stride=(1,1))
        net = slim.conv2d(net,np.rint(capacity*128),[5,1],stride=(1,1))
        net = slim.conv2d(net,2,[8,1],normalizer_fn=None,activation_fn=None)

        net = slim.flatten(tf.reduce_max(net,[1]))

        net = tf.squeeze(net)

        return net 

def network_v2(net, is_training=True, activation_fn=tf.nn.relu, capacity=1.0):

    Nb = net.get_shape()[0]

    net = tf.reshape(net,(-1,2000,200))
    net = tf.expand_dims(net,-1)

    with slim.arg_scope(network_arg_scope(is_training=is_training,
        activation_fn=activation_fn)):

        # extract window features
        net = slim.conv2d(net,np.rint(capacity*8),[1,5],stride=(1,2))
        net = slim.conv2d(net,np.rint(capacity*16),[1,5],stride=(1,2))
        net = slim.conv2d(net,np.rint(capacity*32),[1,5],stride=(1,2))
        net = tf.reduce_max(net,[2],keep_dims=True)

        # combine window features
        net = slim.conv2d(net,np.rint(capacity*64),[5,1],stride=(1,1))
        net = slim.conv2d(net,np.rint(capacity*128),[5,1],stride=(1,1))
        net = slim.conv2d(net,2,[8,1],normalizer_fn=None,activation_fn=None)

        net = slim.flatten(tf.reduce_max(net,[1]))

        net = tf.squeeze(net)

        return net 

def network_v1(net, is_training=True, activation_fn=tf.nn.relu, capacity=1.0):
    # a simple network that works entirely in the temporal domain
    # first step is to avg_pool2d to make it small enough to not break
    # tensorflow.

    net = tf.expand_dims(net,-1)
    net = tf.expand_dims(net,-1)

    with slim.arg_scope(network_arg_scope(is_training=is_training,
        activation_fn=activation_fn)):

        net = slim.avg_pool2d(net,[5,1],stride=(5,1)) 

        net = slim.conv2d(net,np.rint(capacity*16),[5,1],stride=(2,1))
        net = slim.conv2d(net,np.rint(capacity*32),[5,1],stride=(2,1))

        net = slim.conv2d(net,np.rint(capacity*64),[3,1],stride=(2,1))

        net = slim.max_pool2d(net,[11,1],stride=(5,1)) 

        net = slim.conv2d(net,np.rint(capacity*64),[5,1],stride=(2,1))
        net = slim.conv2d(net,np.rint(capacity*128),[5,1],stride=(2,1))
        net = slim.conv2d(net,2,[8,1],normalizer_fn=None,activation_fn=None)

        net = slim.flatten(tf.reduce_max(net,[1]))

        net = tf.squeeze(net)

        return net 

networks = {
        'v4':network_v4,
        'v3':network_v3,
        'v2.3':network_v2_3,
        'v2.2':network_v2_2,
        'v2.1':network_v2_1,
        'v2':network_v2,
        'v1':network_v1
        }

