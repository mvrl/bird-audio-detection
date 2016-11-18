from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import wave
import ops

def load_wav_file(name):
    return data0 

def read_and_decode(recname,is_training=True):


    def read_wav(f):
        try:
            basedir = '/home/nja224/data/birddetection/wav/'
            fid = wave.open(basedir+f+'.wav', "rb")
            raw = fid.readframes(fid.getnframes())
            y = np.fromstring(raw,dtype=np.int16).astype(np.float32)
            y = y / 32768.
            return y
        except Exception,e:
            print(e)

    y = tf.py_func(read_wav, [recname], [tf.float32])

    if is_training:
        # dataset augmentation
        y = tf.reshape(y,(441000,1))
        y = tf.random_crop(y,(400000,1))
        y = tf.squeeze(y)
    else:
        y = tf.reshape(y,(-1,1,1))
        # warblr dataset is variably sized (pad then crop as a lazy # solution)
        y = ops.resize_image_with_crop_or_pad(y,441000,1)
        y = tf.random_crop(y,(400000,1,1)) 
        y = tf.squeeze(y)

    return y

def records(dataset_file,is_training=True):

    fq = tf.train.string_input_producer([dataset_file])
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(fq)
    defaults = [['missing'],[0]]
    recname, label = tf.decode_csv(value,record_defaults=defaults)

    feat = read_and_decode(recname,is_training=is_training)

    tensors = [feat, label, recname]

    if is_training:
        return tf.train.shuffle_batch(tensors, batch_size=64,
                capacity=1000, min_after_dequeue=400)
    else:
        return tf.train.batch(tensors, batch_size=64, num_threads=32)

