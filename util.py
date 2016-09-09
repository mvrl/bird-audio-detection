from __future__ import division, print_function, absolute_import

import os
import tensorflow as tf

def _f(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.tolist()))
def _l(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def export_to_records(filename, features, labels):

    writer = tf.python_io.TFRecordWriter(filename)

    for index in range(labels.size):

        feature={'label': _l(int(labels[index]))}
        for (key,value) in features.items(): 
            feature[key] = _f(value[index,:])

        example = tf.train.Example(
                features=tf.train.Features(
                    feature=feature))

        writer.write(example.SerializeToString())

    writer.close()


