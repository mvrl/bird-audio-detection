from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import wave
import os
import glob
import ops

d = 400000 # number of audio samples for learning

basedir = '../data/'

def read_and_decode(recname):

    def read_wav(f):
        try:
            fid = wave.open(basedir+f+'.wav', "rb")
            raw = fid.readframes(fid.getnframes())
            y = np.fromstring(raw,dtype=np.int16).astype(np.float32)

            # pad if necessary 
            amount_short = 441000-y.size
            if 0 < amount_short:
                y = np.pad(y, 
                        (0,amount_short),
                        'wrap') 

            y = y / 32768.
            #y = y / np.sqrt(1e-8 + np.mean(y**2))
            #y = y / 100.

            return y
        except Exception,e:
            print(e)

    y = tf.py_func(read_wav, [recname], [tf.float32])

    # warblr dataset has variable length audio files
    # pad then crop as a lazy solution

    y = tf.reshape(y,(-1,1,1))

    def fixshort(y):
        # too long, so pad to correct length

        y = ops.resize_image_with_crop_or_pad(y,d,1)
        y = tf.reshape(y,(d,1,1))
        return y

    def fixlong(y):
        # too long, so take a random crop

        # could add the following to force it to be a random crop from
        # near the beginning of the file

        #y = ops.resize_image_with_crop_or_pad(y,441000,1)
        y = tf.random_crop(y,(d,1,1)) 
        return y

    y = tf.cond(
            tf.size(y) <= d,
            lambda: fixshort(y),
            lambda: fixlong(y))

    y = tf.squeeze(y)

    return y

def _loader(name):

    fq = tf.train.string_input_producer([name])
    reader = tf.TextLineReader()
    key, value = reader.read(fq)
    defaults = [['missing'],[0]]
    recname, label = tf.decode_csv(value,record_defaults=defaults)
    feat = read_and_decode(recname)

    return feat, label, recname 

def _get_names(dataset_name, is_training):

    if is_training:
        names = glob.glob('./dataset/%s*0[0-8].csv' % dataset_name)
    else:
        names = glob.glob('./dataset/%s*09.csv' % dataset_name)

    if not names:
        raise Exception('No fold files found.  You probably need to run ./dataset/make_dataset.py')

    return names

def _augment(tensors,augment_add=False,batch_size=1):

    # same audio files, two different shuffles, add together to form
    # new audio files

    feat1, label1, recname1 = tf.train.shuffle_batch_join(
            tensors, batch_size=batch_size,
            capacity=1000, min_after_dequeue=400,
            enqueue_many=True)

    if not augment_add:

        return feat1, label1, recname1

    else:

        feat2, label2, recname2 = tf.train.shuffle_batch_join(
                tensors, batch_size=batch_size,
                capacity=1000, min_after_dequeue=400,
                enqueue_many=True)

        r = tf.random_uniform((batch_size,1))

        feat = r*feat1 + (1-r)*feat2

        # update the label, should not be needed because both labels
        # should be the same
        label = tf.minimum(1,label1 + label2) # element-wise or

        recname = recname1 + '|' + recname2

        return feat, label, recname

def stratRecords(dataset_names=[''],is_training=True,
                 batch_size=64,augment_add=False):
    _records = []

    for dataset_name in dataset_names:
        _records.append(records(dataset_name=dataset_name,
                                is_training=is_training,
                                batch_size=batch_size,
                                augment_add=augment_add))

    feat, label, recname = tf.train.shuffle_batch_join( 
                                            _records,
                                            batch_size=batch_size,
                                            capacity=1000,
                                            min_after_dequeue=400,
                                            enqueue_many=True)
    return feat, label, recname

def records(dataset_name,is_training=True,batch_size=64,augment_add=False):

    names = _get_names(dataset_name, is_training)

    if not is_training:

        tensors = []
        for f in names:
            tensors.append(_loader(f))

        # no need to shuffle test data
        feat, label, recname = tf.train.batch_join(
                tensors, batch_size=batch_size)

    else:

        tensors = []
        for f in names:
            tensors.append(_loader(f))

        feat, label, recname = tf.train.shuffle_batch_join(tensors,
                batch_size=1, capacity=1000,
                min_after_dequeue=400)

        (feat,recname),label = tf.contrib.training.stratified_sample(
            [feat,recname],label,[.5,.5],
            batch_size,
            queue_capacity=300,
            enqueue_many=True) 

    return feat, label, recname
