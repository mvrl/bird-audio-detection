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
        raise Exception("Don't know what to grab, it has to be train, test or all.")

    if not names:
        raise Exception('No fold files found.  You probably need to run ./dataset/make_dataset.py')

    return names

def _augment(tensors, neg_tensors, batch_size=16):

    # SUGGESTION: we used to have code to isolate positives and
    # negatives then we would use the code below to merge only
    # positives and negatives.  probably a better strategy is to just
    # get another stream of negatives and randomly add it in to all
    # the examples

    # same audio files, two different shuffles, add together to form
    # new audio files

    feat, label, recname = tensors
    neg_feat, neg_label, neg_recname = neg_tensors

    r = tf.random_uniform((batch_size,1))

    # r represents the percentage of signal we want to keep
    # It is a positively skewed data series between 0.25 to 1
    r = 1 - 0.25 * tf.log(1. + 100*r) / tf.log(101.)

    feat = r*feat + (1-r)*neg_feat

    # Shouldn't need to update label, as augmentation is an "|" operation

    recname = recname + '|' + neg_recname

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

    if augment_add:

        if not os.path.exists('./dataset/negative_samples.csv'):
            raise Exception("Negative samples not found, run 'cd dataset' and 'python make_dataset.py' first, then train")

        _neg_tensors = _load_tensors('./dataset/negative_samples.csv')
        neg_tensors = tf.train.batch(_neg_tensors, batch_size=batch_size)
        tensors = _augment(tensors, neg_tensors, batch_size=batch_size)

    return tensors 


# Load all of badchallenge files
def records_challenge(dataset_names=['badchallenge'], 
        is_training=False, batch_size=64, augment_add=False):

    if augment_add:
        print('Ignoring augmentation in test mode')

    return _records(dataset_names=dataset_names,
                                what_to_grab='all',
                                num_epochs=1,
                                augment_add=False,
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
        is_training=False, batch_size=64, augment_add=False):

    if augment_add:
        print('Ignoring augmentation add in test mode')

    return _records(dataset_names=dataset_names,
                                what_to_grab='test',
                                num_epochs=1,
                                augment_add=False,
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

