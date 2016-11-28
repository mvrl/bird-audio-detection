# TODO add filename and line number to TF Record

from __future__ import division, print_function, absolute_import

import os
from optparse import OptionParser
import tensorflow as tf
import ops

ttf = tf.train.Feature
_b = lambda v: ttf(bytes_list=tf.train.BytesList(value=v))
_f = lambda v: ttf(float_list=tf.train.FloatList(value=v.tolist()))
_l = lambda v: ttf(int64_list=tf.train.Int64List(value=v))

def export_to_records(filename, dat, features, labels1, labels2):

    writer = tf.python_io.TFRecordWriter(filename,
            options=tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB))

    for index in range(labels1.size):

        if labels1[index] not in [1,2,3] or labels2[index] not in [1,2,3]:
            continue

        feature={
                'file': _b([dat]),
                'row':_l([index]),
                'label': _l([int(labels1[index]), int(labels2[index])])
            }

        for (key,value) in features.items(): 
            feature[key] = _f(value[index,:])

        example = tf.train.Example(
                features=tf.train.Features(
                    feature=feature))

        writer.write(example.SerializeToString())

    writer.close()


def parse_arguments():

    parser = OptionParser()

    parser.add_option("-a", "--activation", dest="AFN", default='relu')
    parser.add_option("-n", "--network", dest="NET", default='v2')
    #parser.add_option("-f", "--feature", dest="FEAT", default='eeg')
    parser.add_option("-c", "--capacity", dest="CAP", type="float", default=1.0)

    (opts, args) = parser.parse_args()

    nc = {}
    nc['activation_fn'] = {
        'elu':tf.nn.elu,
        'relu':tf.nn.relu,
        'lrelu':ops.lrelu
        }[opts.AFN]

    nc['network'] = opts.NET

    nc['capacity'] = opts.CAP

    return nc

def run_name(nc):

    # define run name
    run_name = nc['network']
    run_name += '_' + nc['activation_fn'].func_name

    run_name += '_{:0.2f}'.format(nc['capacity'])

    return run_name

