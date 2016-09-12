from __future__ import division, print_function, absolute_import

import os
import tensorflow as tf
import numpy as np
import dataset
import network

slim = tf.contrib.slim

#
#
#

print('Setting up run')

use_eeg = True

run_name = 'elu'
if use_eeg:
    run_name += '_eeg'
else:
    run_name += '_piezo'

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint_dir', 'checkpoint/' + run_name + '/','output directory for model checkpoints')

#
# Define graph 
#

with tf.variable_scope('Input'):
    print('Defining input pipeline')

    features, label1, label2 = dataset.records('test.txt',
            use_eeg=use_eeg,
            is_training=False)

    # why is this necessary?
    label1 = tf.reshape(label1,[-1])
    label2 = tf.reshape(label2,[-1])

    label = tf.concat(0,(label1,label2))

with tf.variable_scope('Predictor'):
    print('Defining prediction network')

    logits = network.network(features, use_eeg=use_eeg,
            is_training=False)

    # replicate because we have two annotaters
    logits = tf.concat(0,(logits,logits))

    prediction = tf.argmax(logits,1)

    acc = tf.contrib.metrics.accuracy(prediction,label)
    conf = tf.contrib.metrics.confusion_matrix(prediction,label,num_classes=tf.cast(3,tf.int64),dtype=tf.int64)

with tf.Session() as sess:
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    sess.run(tf.initialize_all_variables())

    saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path: 
        print('Restoring checkpoint')
        saver.restore(sess, ckpt.model_checkpoint_path)

    print('Starting evaluation')
    _conf_accum = np.zeros((3,3), dtype=np.int64)
    for ix in xrange(10000):

        _conf,_acc = sess.run([conf,acc])

        print('Accuracy = {}'.format(_acc))
        _conf_accum += _conf

        if ix % 10 == 0:
            print(_conf_accum)# / np.sum(_conf_accum))

    coord.request_stop()
    coord.join(threads)

