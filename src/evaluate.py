#
# Purpose: this file generates predictions for us to use internally
# to evaluate different learning methods on the training data.
#

from __future__ import division, print_function, absolute_import

import os
import sys
import tensorflow as tf
import numpy as np
import dataset
import network
import itertools
import util

slim = tf.contrib.slim

#
# parse inputs 
#

nc,dc = util.parse_arguments()
run_name = util.run_name(nc,dc)

checkpoint_dir = 'checkpoint/' + run_name + '/'
out_file = checkpoint_dir + 'output.csv'

if os.path.isfile(out_file):
    print('Skipping ({:s}): output file ({:s}) already exists'.format(run_name, out_file))
    sys.exit(0) 

#
# Define graph 
#

with tf.variable_scope('Input'):
    print('Defining input pipeline')

    feat, label, recname = dataset.records_test_fold(**dc)

with tf.variable_scope('Predictor'):
    print('Defining prediction network')

    logits = network.network(feat,is_training=False,**nc)

    probs = tf.nn.softmax(logits)
    prediction = tf.cast(tf.argmax(logits,1),dtype=tf.int32)

    acc, acc_up = tf.contrib.metrics.streaming_accuracy(prediction,label)
    auc, auc_up = tf.contrib.metrics.streaming_auc(probs[:,1],label)
    conf = tf.contrib.metrics.confusion_matrix(prediction,label,num_classes=tf.cast(2,tf.int64),dtype=tf.int64)

#
# Setup runtime and process 
#

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    sess.run(tf.initialize_local_variables())
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path: 
        print('Restoring checkpoint')
        saver.restore(sess, ckpt.model_checkpoint_path)

    print('Starting evaluation')
    _conf_accum = np.zeros((2,2), dtype=np.int64)

    try:
        with open(out_file,'w') as output:

            for i in itertools.count(0):

                _conf,_acc,_auc,_,_,_prob,_label,_recname = \
                sess.run([conf,acc,auc,acc_up,auc_up,probs,label,recname])

                np.savetxt(output,
                        np.concatenate((_label.reshape((-1,1)),_prob),axis=1),
                        fmt='%u %1.8f %1.8f') 

                print('Accuracy = {0:.3f} AUC = {1:.3f}'.format(_acc,_auc))
                _conf_accum += _conf

                # dump activations 
                #if ix == 0:
                #    print('Activations')
                #    mv = tf.get_collection(tf.GraphKeys.ACTIVATIONS)
                #    for v in mv:
                #        print(v.name)
                #        print(v.outputs.get_shape().as_list()[1:])

                if i % 10 == 0:
                    print(_conf_accum)

    except tf.errors.OutOfRangeError:
        print('Queue empty, exiting now...')
    finally:
        coord.request_stop()
        coord.join(threads)

