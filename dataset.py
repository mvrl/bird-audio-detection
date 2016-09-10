from __future__ import division, print_function, absolute_import

import tensorflow as tf

def read_and_decode(dataset_file):

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(dataset_file)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'Piezo': tf.FixedLenFeature([1600,1,1],tf.float32),
            'EEG1': tf.FixedLenFeature([1600,1,1],tf.float32),
            'EEG2': tf.FixedLenFeature([1600,1,1],tf.float32),
            'EMG': tf.FixedLenFeature([1600,1,1],tf.float32),
            'label': tf.FixedLenFeature([1,2], tf.int64) # there were two annotaters
        })

    piezo = 2*(features['Piezo'] - 2.47)
    eeg1 = features['EEG1'] / 30.
    eeg2 = features['EEG2'] / 80.
    emg = features['EMG'] / 30.

    feature = tf.concat(2,(eeg1,eeg2,emg))

    label1, label2 = tf.split(1,2,features['label'] - 1)

    label1 = tf.reshape(label1,[-1])
    label2 = tf.reshape(label2,[-1])

    return (feature, label1, label2) 

def records(dataset_file,datadir='./records/'):

    qs = []
    for d in [tmp.strip() for tmp in open(dataset_file, 'r')]:
        fq = tf.train.string_input_producer([datadir + d + '.tfrecord'])
        qs.append(read_and_decode(fq))

    #return qs
    return tf.train.shuffle_batch_join( qs, batch_size=512,
            capacity=1000000, min_after_dequeue=1000)


