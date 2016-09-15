# TODO include dataset name and record number in the output file so we
# can compare across datasets

from __future__ import division, print_function, absolute_import

import os
import tensorflow as tf
import numpy as np
import dataset
import network
import util
slim = tf.contrib.slim

#
#
#

print('Setting up run')
nc = util.parse_arguments()
run_name = util.run_name(nc)

use_eeg = False 

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint_dir', 'checkpoint/' + run_name + '/','output directory for model checkpoints')

#
# Define graph 
#

with tf.variable_scope('Input'):
    print('Defining input pipeline')

    features, label1, label2, data_row, fileid = dataset.records('test.txt',
            use_eeg=nc['use_eeg'],
            is_training=False)

    label_output = tf.concat(1,(label1,label2))

    # why is this necessary?
    label1 = tf.reshape(label1,[-1])
    label2 = tf.reshape(label2,[-1])

    label = tf.concat(0,(label1,label2))

with tf.variable_scope('Predictor'):
    print('Defining prediction network')

    logits = network.network(features,
            is_training=False,
            use_eeg=nc['use_eeg'],
            activation_fn=nc['activation_fn'])

    probs = tf.nn.softmax(logits)

    # replicate because we have two annotaters
    logits = tf.concat(0,(logits,logits))

    prediction = tf.argmax(logits,1)

    acc = tf.contrib.metrics.accuracy(prediction,label)
    conf = tf.contrib.metrics.confusion_matrix(prediction,label,num_classes=tf.cast(3,tf.int64),dtype=tf.int64)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    
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
    with open(FLAGS.checkpoint_dir + '/output.csv','w') as output:

        for ix in xrange(10000):

            _conf,_acc,_prob,_label_output,_data_row,_fileid = \
               sess.run([conf,acc,probs,label_output,data_row,fileid])

            _fileid = np.array([int(f[2:]) for f in _fileid]).reshape([-1,1])

            np.savetxt(output,
                    np.concatenate((_fileid,_data_row,_prob,_label_output+1),axis=1),
                    fmt='%u %u %1.8f %1.8f %1.8f %u %u') 

            print('Accuracy = {}'.format(_acc))
            _conf_accum += _conf

            # dump activations 
            if ix == 0:

                print('Activations')
                mv = tf.get_collection(tf.GraphKeys.ACTIVATIONS)
                for v in mv:
                    print(v.name)
                    print(v.outputs.get_shape().as_list()[1:])

            if ix % 10 == 0:
                print(_conf_accum)# / np.sum(_conf_accum))

    coord.request_stop()
    coord.join(threads)

