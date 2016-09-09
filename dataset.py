from __future__ import division, print_function, absolute_import

import tensorflow as tf

def read_and_decode(dataset_file):

    reader = tf.TFRecordReader()

    wack, serialized_example = reader.read(dataset_file)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'Piezo': tf.FixedLenFeature([800,],tf.float32),
            'EEG1': tf.FixedLenFeature([800,],tf.float32),
            'EEG2': tf.FixedLenFeature([800,],tf.float32),
            'EMG': tf.FixedLenFeature([800,],tf.float32),
            'label': tf.FixedLenFeature([1,], tf.int64),
        })

    piezo = features['Piezo']
    label = features['label']

    return (wack, piezo, label) 

def records(dataset_file,datadir='./records/'):

    qs = []
    for d in [tmp.strip() for tmp in open(dataset_file, 'r')]:
        fq = tf.train.string_input_producer([datadir + d + '.tfrecord'])
        qs.append(read_and_decode(fq))

    #return qs
    return tf.train.shuffle_batch_join( qs, batch_size=64, capacity=1000000, min_after_dequeue=5000)


