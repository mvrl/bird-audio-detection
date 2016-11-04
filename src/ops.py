import tensorflow as tf

def lrelu(x,leak=.2):
    return tf.maximum(x,leak*x)

