import tensorflow as tf
slim = tf.contrib.slim

def network_arg_scope(
        weight_decay=0.00004,
        is_training=True,
        batch_norm_var_collection='moving_vars'):

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

    lrelu = lambda x: tf.maximum(.1*x,x)
    afn = lrelu
    #afn = tf.nn.relu
    #afn = tf.nn.elu

    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope([slim.conv2d],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        weights_initializer=slim.variance_scaling_initializer(),
        padding='VALID',
        outputs_collections=tf.GraphKeys.ACTIVATIONS,
        activation_fn=afn,
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.max_pool2d], 
            stride=(2,1),
            outputs_collections=tf.GraphKeys.ACTIVATIONS):
            with slim.arg_scope([slim.batch_norm], is_training=is_training) as sc:
                return sc

def network(net, is_training=True, use_eeg=True):
    with slim.arg_scope(network_arg_scope(is_training=is_training)):

        net = slim.conv2d(net,16,[5,1],stride=(2,1))
        net = slim.conv2d(net,32,[5,1],stride=(2,1))

        if use_eeg:
            net = slim.conv2d(net,64,[3,3],stride=(2,1))
        else:
            net = slim.conv2d(net,64,[3,1],stride=(2,1))

        net = slim.max_pool2d(net,[11,1],stride=(4,1)) 
        net = slim.conv2d(net,64,[5,1],stride=2)
        net = slim.conv2d(net,128,[5,1],stride=2)
        print(net.get_shape().as_list())
        net = slim.conv2d(net,3,[8,1],normalizer_fn=None,activation_fn=None)

        net = slim.flatten(tf.reduce_mean(net,[1]))

        #print(net.get_shape().as_list())

        #net = slim.conv2d(net,16,[7,1],stride=2)
        #net = slim.conv2d(net,32,[7,1],stride=2)
        #net = slim.conv2d(net,64,[7,1],stride=2)
        #net = slim.conv2d(net,3,[7,1],stride=2)

        return net 

