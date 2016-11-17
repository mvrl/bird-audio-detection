from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import librosa

def read_and_decode(recname,is_training=True):

    #if is_training:
    #    # dataset augmentation
    #    offset = np.random.randint(0,100)
    #else:
    #    offset = 99

    #feature = feature[offset:(offset+1500),:]

    def read_wav(f):
        basedir = '/u/eag-d1/scratch/jacobs/birddetection/wav/'
        y, sr = librosa.load(basedir + f + '.wav')
        return y

    (y,) = tf.py_func(read_wav, [recname], [tf.float32])
    y = tf.reshape(y,(220500,))

    return y

def records(dataset_file,is_training=True):

    fq = tf.train.string_input_producer([dataset_file])
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(fq)
    defaults = [['missing'],[0]]
    recname, label= tf.decode_csv(value,record_defaults=defaults)

    feat = read_and_decode(recname)

    tensors = [feat, label, recname]

    if is_training:
        return tf.train.shuffle_batch(tensors, batch_size=10,
                capacity=100, min_after_dequeue=50)
    else:
        return tf.train.batch(tensors, batch_size=20)


