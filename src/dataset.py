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
            y = y / 32768.
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

def _loader(names):

    fq = tf.train.string_input_producer(names)
    reader = tf.TextLineReader()
    key, value = reader.read(fq)
    defaults = [['missing'],[0]]
    recname, label = tf.decode_csv(value,record_defaults=defaults)
    feat = read_and_decode(recname)

    return feat, label, recname 

def _get_names(is_training):

    if is_training:
        names = glob.glob('./dataset/*0[0-8].csv')
    else:
        names = glob.glob('./dataset/*09.csv')

    if not names:
        raise Exception('No fold files found.  You probably need to run ./dataset/make_dataset.py')

    return names

def _augment(tensors,batch_size=16):

    # same audio files, two different shuffles, add together to form
    # new audio files

    feat1, label1, recname1 = tf.train.shuffle_batch_join(
            tensors, batch_size=batch_size,
            capacity=1000, min_after_dequeue=400,
            enqueue_many=True)

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

def records(is_training=True,batch_size=64,augment_add=False):

    names = _get_names(is_training)

    if not augment_add:

        tensors = []
        for f in names:
            tensors.append(_loader(names))

        if is_training:
            # randomly shuffle the examples
            feat, label, recname = tf.train.shuffle_batch_join(
                    tensors, batch_size=batch_size, capacity=1000,
                    min_after_dequeue=400)
        else:
            # no need to shuffle test data
            feat, label, recname = tf.train.batch_join(
                    tensors, batch_size=batch_size)
    else:

        # this is a special version of training that does data
        # augmentation by adding two audio files together

        tensors_neg = []
        tensors_pos = []

        for f in names:

            # randomly shuffle, and randomly add together sounds to
            # make new sounds.  Idea here is to only add positives to
            # positives and negatives to negatives

            feat, label, recname = _loader(names)

            # add batch dimension to support filtering by label
            feat = tf.expand_dims(feat,0)
            recname = tf.expand_dims(recname,0)
            recname = tf.expand_dims(recname,0)
            tmp = label
            label = tf.expand_dims(label,0)

            # only negatives
            feat_neg = feat[0:(1-tmp),:]
            recname_neg = recname[0:(1-tmp),:]
            label_neg = label[0:(1-tmp)]

            # only positives
            feat_pos = feat[0:(tmp),:]
            recname_pos = recname[0:(tmp),:]
            label_pos = label[0:(tmp)]

            tensors_neg.append((feat_neg, label_neg, recname_neg))
            tensors_pos.append((feat_pos, label_pos, recname_pos))

        pos = _augment(tensors_pos)
        neg = _augment(tensors_neg)

        feat, label, recname = tf.train.shuffle_batch_join((pos,neg),
                batch_size=batch_size, capacity=1000,
                min_after_dequeue=400, enqueue_many=True)

    return feat, label, recname

