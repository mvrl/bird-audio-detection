from __future__ import division, print_function, absolute_import

import os
from optparse import OptionParser
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


def parse_arguments():

    parser = OptionParser()

    parser.add_option("-a", "--activation", dest="AFN",default='relu')
    parser.add_option("-f", "--feature", dest="FEAT",default='eeg')

    (opts, args) = parser.parse_args()

    nc = {}
    nc['activation_fn'] = {
        'elu':tf.nn.elu,
        'relu':tf.nn.relu
        }[opts.AFN]

    nc['use_eeg'] = {
        'eeg':True,
        'piezo':False
        }[opts.FEAT]

    return nc

def run_name(nc):

    # define run name
    run_name = nc['activation_fn'].func_name

    if nc['use_eeg']:
        run_name += '_eeg'
    else:
        run_name += '_piezo'

    return run_name

