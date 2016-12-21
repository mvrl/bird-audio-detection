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
    y = tf.reshape(y,(-1,1,1))
    y = tf.random_crop(y,(d,1,1)) 
    y = tf.squeeze(y)

    return y 

def _load_tensors(name, num_epochs=None):

    fq = tf.train.string_input_producer([name],num_epochs=num_epochs)
    reader = tf.TextLineReader()
    key, value = reader.read(fq)
    defaults = [['missing'],[0]]
    recname, label = tf.decode_csv(value,record_defaults=defaults)
    feat = read_and_decode(recname) 

    return feat, label, recname 

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
            capacity=1000, min_after_dequeue=400)

    feat2, label2, recname2 = tf.train.shuffle_batch_join(
            tensors, batch_size=batch_size,
            capacity=1000, min_after_dequeue=400)

    r = tf.random_uniform((batch_size,1))

    feat = r*feat1 + (1-r)*feat2

    # update the label, should not be needed because both labels
    # should be the same
    label = tf.minimum(1,label1 + label2) # element-wise or

    recname = recname1 + '|' + recname2

    return feat, label, recname

def _records(dataset_names=[''], what_to_grab='train', is_training=True, 
                    batch_size=64, augment_add=False, num_epochs=None):

    # Grab all desired folds
    names = []
    for dataset_name in dataset_names:
        names.extend(_get_names(dataset_name, what_to_grab=what_to_grab))
    
    tensors_list = []
    for f in names:
        tensors = _load_tensors(f,num_epochs=num_epochs)
        if augment_add:
            tensors = _augment(tensors)
        tensors_list.append(tensors)

    if is_training:
        tensors = tf.train.shuffle_batch_join(
                                    tensors_list,
                                    batch_size=batch_size,
                                    capacity=1000,
                                    min_after_dequeue=400)
    else:
        # no need to shuffle test data
        tensors = tf.train.batch_join(tensors_list, 
                                    batch_size=batch_size,
                                    allow_smaller_final_batch=True)

    return tensors 


# Load all of badchallenge files
def records_challenge(dataset_names=['badchallenge'], is_training=False, 
                      batch_size=64):

    return _records(dataset_names=dataset_names,
                                what_to_grab='all',
                                num_epochs=1,
                                is_training=is_training,
                                batch_size=batch_size)

# Load all of ff and warblr, 0-8 only
def records_train_fold(dataset_names=['freefield1010', 'warblr'], 
                       is_training=True, batch_size=64, augment_add=False):

    return _records(dataset_names=dataset_names,
                                what_to_grab='train',
                                is_training=is_training,
                                batch_size=batch_size,
                                augment_add=augment_add)

# Load all of ff and warblr, 9 only
def records_test_fold(dataset_names=['freefield1010', 'warblr'], 
                      is_training=False, batch_size=64):

    return _records(dataset_names=dataset_names,
                                what_to_grab='test',
                                num_epochs=1,
                                is_training=is_training,
                                batch_size=batch_size)

# Load all of ff and warblr, 0-9
def records_train_all(dataset_names=['freefield1010', 'warblr'], 
                      is_training=True, batch_size=64, augment_add=False):

    return _records(dataset_names=dataset_names,
                                what_to_grab='all',
                                is_training=is_training,
                                batch_size=batch_size,
                                augment_add=augment_add)

