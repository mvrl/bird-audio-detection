# MAYBE explicitly name layers

import tensorflow as tf
slim = tf.contrib.slim
import numpy as np

def network_arg_scope(
        weight_decay=0.00004,
        is_training=True,
        batch_norm_var_collection='moving_vars',
        activation_fn=tf.nn.relu):

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

def network(net, is_training=True, activation_fn=tf.nn.relu, capacity=1.0):

    net = tf.expand_dims(net,-1)
    net = tf.expand_dims(net,-1)

    with slim.arg_scope(network_arg_scope(is_training=is_training,
        activation_fn=activation_fn)):

        net = slim.avg_pool2d(net,[5,1],stride=(2,1)) 

        net = slim.conv2d(net,np.rint(capacity*16),[5,1],stride=(2,1))
        net = slim.conv2d(net,np.rint(capacity*32),[5,1],stride=(2,1))

        net = slim.conv2d(net,np.rint(capacity*64),[3,1],stride=(2,1))

        net = slim.max_pool2d(net,[11,1],stride=(4,1)) 
        net = slim.conv2d(net,np.rint(capacity*64),[5,1],stride=(2,1))
        net = slim.conv2d(net,np.rint(capacity*128),[5,1],stride=(2,1))
        net = slim.conv2d(net,2,[8,1],normalizer_fn=None,activation_fn=None)

        print(net.get_shape().as_list())
        net = slim.flatten(tf.reduce_mean(net,[1]))

        #print(net.get_shape().as_list())

        #net = slim.conv2d(net,16,[7,1],stride=2)
        #net = slim.conv2d(net,32,[7,1],stride=2)
        #net = slim.conv2d(net,64,[7,1],stride=2)
        #net = slim.conv2d(net,3,[7,1],stride=2)

        net = tf.squeeze(net)

        return net 

