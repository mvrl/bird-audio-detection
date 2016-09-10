import tensorflow as tf
slim = tf.contrib.slim

def network_arg_scope(weight_decay=0.00004,
                           batch_norm_var_collection='moving_vars'):
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': 0.9997,
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

  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.conv2d],
                      weights_regularizer=slim.l2_regularizer(weight_decay)):
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=slim.variance_scaling_initializer(),
        padding='VALID',
        activation_fn=tf.nn.relu,
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params) as sc:
      return sc

def network(net):

   with slim.arg_scope(network_arg_scope()):

       net = slim.conv2d(net,8,[3,1],stride=2)
       net = slim.conv2d(net,16,[3,1],stride=2)
       net = slim.conv2d(net,16,[3,1],stride=2)
       net = slim.conv2d(net,32,[3,1],stride=2)
       net = slim.conv2d(net,32,[3,1],stride=2)
       net = slim.conv2d(net,64,[3,1],stride=2)
       net = slim.conv2d(net,16,[15,1],stride=2)
       net = slim.conv2d(net,3,[3,1])
       #net = slim.conv2d(net,16,[7,1],stride=2)
       #net = slim.conv2d(net,32,[7,1],stride=2)
       #net = slim.conv2d(net,64,[7,1],stride=2)
       #net = slim.conv2d(net,3,[7,1],stride=2)
       print(net.get_shape().as_list())
       net = slim.flatten(tf.reduce_mean(net,[1]))

       return net 




