import tensorflow as tf

def lrelu(x,leak=.2):
    return tf.maximum(x,leak*x)

# Adapted from https://github.com/tensorflow/tensorflow/issues/1246
def spatial_norm(tensor, height):

    bias = tf.constant(0.000001, dtype=tf.float32)

    squared = tf.square(tensor)

    squared_mean = tf.nn.avg_pool(squared,
                            [1, height, 1, 1],
                            [1, 1, 1, 1],
                            padding='SAME')

    return tensor / ((0.000001 + squared_mean) ** 0.5), squared_mean

