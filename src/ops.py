import tensorflow as tf

def lrelu(x,leak=.2):
    return tf.maximum(x,leak*x,name="LReLU")

# Adapted from https://github.com/tensorflow/tensorflow/issues/1246
def temporal_norm(tensor, height):
    '''Normalizes along axis=1 (time) for a 4D input tensor'''

    w = tensor.get_shape()[2]

    squared = tf.square(tensor)
    squared = tf.reduce_mean(squared,axis=2,keep_dims=True)
    squared_mean = tf.nn.avg_pool(squared,
                            [1, height, 1, 1],
                            [1, 1, 1, 1],
                            padding='SAME')

    normed = tf.div(tensor, (0.000001 + squared_mean) ** 0.5)

    return normed, tf.tile(squared_mean,(1, 1, w, 1))

