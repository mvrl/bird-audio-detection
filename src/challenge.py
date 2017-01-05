#
# Purpose: this file generates predictions for us to submit to the
# bird audio detection challenge server.
#

from __future__ import division, print_function, absolute_import

import os
import sys
import tensorflow as tf
import numpy as np
import dataset
import network
import util
import collections
slim = tf.contrib.slim

#
#
#

nc,dc,rc = util.parse_arguments()
run_name = util.run_name(nc,dc,rc)

checkpoint_dir = 'checkpoint/' + run_name + '/'
out_file = checkpoint_dir + 'submission.csv'

if os.path.isfile(out_file):
    print('Skipping ({:s}): output file ({:s}) already exists'.format(run_name, out_file))
    sys.exit(0) 

#
# Define graph 
#

with tf.variable_scope('Input'):
    print('Defining input pipeline')

    num_epochs = 30 

    feat, label, recname = dataset.records_challenge(num_epochs=num_epochs, **dc)

with tf.variable_scope('Predictor'):
    print('Defining prediction network')

    logits = network.network(feat,is_training=False,**nc)

    probs = tf.nn.softmax(logits)
    score = probs[:,1]

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    sess.run(tf.local_variables_initializer())
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path: 
        print('Restoring checkpoint')
        saver.restore(sess, ckpt.model_checkpoint_path)

    print('Starting evaluation')

    _scores = collections.defaultdict(float)

    try:

        while(True):

            _score,_recname = sess.run([score,recname])

            print('Processed batch of size %i with average score %f'%(_score.size,_score.mean()))

            for idx, name in enumerate(_recname):
                _scores[name] += _score[idx]

    except tf.errors.OutOfRangeError:
        print('Queue empty, exiting now...')
    finally:
        coord.request_stop()
        coord.join(threads)

    print('Exporting to %s.'%out_file)
    with open(out_file,'w') as fid:
        print("itemid,hasbird",file=fid)
        for key in sorted(_scores):
            out_name = key.split('/')[2]
            print("%s,%1.8f" % (out_name, _scores[key] / num_epochs),file=fid)
