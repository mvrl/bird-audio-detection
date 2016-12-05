from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import wave
import os
import glob
import ops

d = 400000 # number of audio samples for learning

basedir = os.path.expanduser('../data/')

def read_and_decode(recname,is_training=True):


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

def records(is_training=True,batch_size=64,exclude_positive=False,augment_with_negatives=False):

    if is_training:
        names = glob.glob('./dataset/*_train.csv')
    else:
        names = glob.glob('./dataset/*_test.csv')

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

        #
        # filter by label (if requested)
        #

        # add batch dimension to support filtering by label
        feat = tf.expand_dims(feat,0)
        recname = tf.expand_dims(recname,0)
        recname = tf.expand_dims(recname,0)
        tmp = label
        label = tf.expand_dims(label,0)

        if exclude_positive:

            feat = feat[0:(1-tmp),:]
            recname = recname[0:(1-tmp),:]
            label = label[0:(1-tmp)]

        tensors.append((feat, label, recname))

    if is_training:

        feat, label, recname = tf.train.shuffle_batch_join(tensors, batch_size=batch_size,
                    capacity=1000, min_after_dequeue=400,
                    enqueue_many=True)

        if augment_with_negatives:

            # idea: add in a negative example to decrease the signal
            # to noise ratio, but also reduce overfitting and help
            # generalization

            # load training examples, but filter out positive examples
            # if we don't do this, then we get way to many positives
            # after we combine them below 
            feat_noise,label_noise,recname_noise = records(
                    is_training=is_training,
                    augment_with_negatives=False,
                    batch_size=batch_size,
                    exclude_positive=True) # only use non bird audio as noise 

            # combine the two audio files
            # MAYBE it might help to randomize this a bit
            feat = .9*feat + .1*feat_noise

            # update the label, currently not needed because we
            # label_noise should always be negative
            label = tf.minimum(1,label + label_noise) # element-wise or

            recname = recname + '|' + recname_noise 

        return feat, label, recname

    else:
        return tf.train.batch_join(tensors, batch_size=batch_size,
                enqueue_many=True)

