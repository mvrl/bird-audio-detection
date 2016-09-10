from __future__ import division, print_function, absolute_import

import os
import tensorflow as tf

def _f(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.tolist()))
def _l(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def export_to_records(filename, features, labels1, labels2):

    writer = tf.python_io.TFRecordWriter(filename)

    for index in range(labels1.size):

        if labels1[index] not in [1,2,3] or labels2[index] not in [1,2,3]:
            continue

        feature={'label': _l([int(labels1[index]),
            int(labels2[index])])}
        for (key,value) in features.items(): 
            feature[key] = _f(value[index,:])

        example = tf.train.Example(
                features=tf.train.Features(
                    feature=feature))

        writer.write(example.SerializeToString())

    writer.close()


