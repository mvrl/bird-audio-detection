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

def _load_csv(name, num_epochs=None):

    fq = tf.train.string_input_producer([name], num_epochs=num_epochs)
    reader = tf.TextLineReader()
    key, value = reader.read(fq)
    defaults = [['missing'],[0]]
    recname, label = tf.decode_csv(value,record_defaults=defaults)

    return label, recname 

def _get_names(dataset_name, what_to_grab='train'):

    if what_to_grab == 'train':
        names = glob.glob('./dataset/%s*0[0-8].csv' % dataset_name)
    
    elif what_to_grab == 'test':
        names = glob.glob('./dataset/%s*09.csv' % dataset_name)

    elif what_to_grab == 'all':
        names = glob.glob('./dataset/%s*0[0-9].csv' % dataset_name)

    else:
        raise Exception("Don't know what to grab, it has to be train, \
                         test or all.")

    if not names:
        raise Exception('No fold files found.  You probably need to run \
                         ./dataset/make_dataset.py')

    return names

def _augment(tensors,batch_size=16):

    raise(Exception('augmentation is not working right now.'))
    # SUGGESTION: we used to have code to isolate positives and
    # negatives then we would use the code below to merge only
    # positives and negatives.  probably a better strategy is to just
    # get another stream of negatives and randomly add it in to all
    # the examples

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

# Load all of badchallenge files
def records_challenge(dataset_names=['badchallenge'], is_training=False, 
                      batch_size=64, augment_add=False):

    feat, label, recname = testRecords(dataset_names=dataset_names,
                                       what_to_grab='all',
                                       batch_size=batch_size)

    return feat, label, recname

# Load all of ff and warblr, 0-8 only
def records_train_fold(dataset_names=['freefield1010', 'warblr'], 
                       is_training=True, batch_size=64, augment_add=False):
    _records = []

    for dataset_name in dataset_names:
        _records.append(stratifyRecords(dataset_name=dataset_name,
                                        what_to_grab='train',
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

# Load all of ff and warblr, 9 only
def records_test_fold(dataset_names=['freefield1010', 'warblr'], 
                      is_training=False, batch_size=64):

    feat, label, recname = testRecords(dataset_names=dataset_names,
                                       what_to_grab='test',
                                       batch_size=batch_size)

    return feat, label, recname

# Load all of ff and warblr, 0-9
def records_train_all(dataset_names=['freefield1010', 'warblr'], 
                      is_training=True, batch_size=64, augment_add=False):
    _records = []

    for dataset_name in dataset_names:
        _records.append(stratifyRecords(dataset_name=dataset_name,
                                        what_to_grab='all',
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

def testRecords(dataset_names=[''], what_to_grab='test', 
                is_training=False, batch_size=64):

    names = []
    # Grab all desired folds
    for dataset_name in dataset_names:
        names.extend(_get_names(dataset_name, what_to_grab=what_to_grab))

    tensors = []
    for f in names:
        label, recname = _load_csv(f, num_epochs=1)
        feat = read_and_decode(recname)
        tensors.append((feat, label, recname))
    
    # no need to shuffle test data
    feat, label, recname = tf.train.batch_join(tensors, 
                                         batch_size=batch_size,
                                         allow_smaller_final_batch=True)
    return feat, label, recname


def stratifyRecords(dataset_name='', what_to_grab='train', is_training=True, 
                    batch_size=64, augment_add=False):

    names = _get_names(dataset_name, what_to_grab=what_to_grab)
    
    tensors = []
    for f in names:
        tensors.append(_load_csv(f, num_epochs=None))

    label, recname = tf.train.batch_join(tensors, batch_size=batch_size)

    (recname,), label = tf.contrib.training.stratified_sample(
        [recname],label,[.5,.5],
        batch_size=batch_size,
        queue_capacity=300,
        enqueue_many=True) 

    feat = tf.pack([read_and_decode(x) for x in tf.unstack(recname)])

    if augment_add:
        feat,label,recname = _augment((feat,label,recname))

    return feat, label, recname
