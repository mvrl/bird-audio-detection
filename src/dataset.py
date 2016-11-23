from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import wave
import glob
import ops

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

    # warblr dataset has variable length audio files
    # pad then crop as a lazy solution
    y = tf.reshape(y,(-1,1,1))
    y = ops.resize_image_with_crop_or_pad(y,441000,1)
    y = tf.random_crop(y,(400000,1,1)) 
    y = tf.squeeze(y)

    return y

def records(is_training=True):

    if is_training:
        names = glob.glob('./dataset/*0[0-8]')
    else:
        names = glob.glob('./dataset/*09')

    if not names:
        raise Exception('No fold files found.  You probably need to run ./dataset/make_dataset.sh')

    tensors = []
    for f in names:

        fq = tf.train.string_input_producer(names)
        reader = tf.TextLineReader()
        key, value = reader.read(fq)
        defaults = [['missing'],[0]]
        recname, label = tf.decode_csv(value,record_defaults=defaults)

        feat = read_and_decode(recname,is_training=is_training)

        tensors.append((feat, label, recname))

    if is_training:
        # TODO make two of these and randomly join them as a form of
        # data augmentation
        return tf.train.shuffle_batch_join(tensors, batch_size=64,
                capacity=1000, min_after_dequeue=400)
    else:
        return tf.train.batch_join(tensors, batch_size=64, num_threads=32)

