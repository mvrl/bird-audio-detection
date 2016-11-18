from __future__ import division, print_function, absolute_import

import os
import sys
import tensorflow as tf
import numpy as np
import dataset
import network
import util
slim = tf.contrib.slim

#
#
#

nc = util.parse_arguments()
run_name = util.run_name(nc)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint_dir', 'checkpoint/' + run_name + '/','output directory for model checkpoints')

out_file = FLAGS.checkpoint_dir + 'output.csv'

if os.path.isfile(out_file):
    print('Skipping ({:s}): output file ({:s}) already exists'.format(run_name, out_file))
    sys.exit(0) 

#
# Define graph 
#

with tf.variable_scope('Input'):
    print('Defining input pipeline')

    feat, label, recname = dataset.records(
            '/home/nja224/data/birddetection/ff1010bird_metadata.csv',
            is_training=False)

with tf.variable_scope('Predictor'):
    print('Defining prediction network')

    logits = network.network(feat,is_training=False,**nc)

    probs = tf.nn.softmax(logits)
    prediction = tf.cast(tf.argmax(logits,1),dtype=tf.int32)

    acc = tf.contrib.metrics.accuracy(prediction,label)
    conf = tf.contrib.metrics.confusion_matrix(prediction,label,num_classes=tf.cast(2,tf.int64),dtype=tf.int64)

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
    _conf_accum = np.zeros((2,2), dtype=np.int64)

    with open(out_file,'w') as output:

        for ix in xrange(10000):

            _conf,_acc,_prob,_label = \
               sess.run([conf,acc,probs,label])

            #_fileid = np.array([int(f[2:]) for f in _fileid]).reshape([-1,1])

            np.savetxt(output,
                    np.concatenate((_label.reshape((-1,1)),_prob),axis=1),
                    fmt='%u %1.8f %1.8f') 

            print('Accuracy = {}'.format(_acc))
            _conf_accum += _conf

            # dump activations 
            #if ix == 0:
            #    print('Activations')
            #    mv = tf.get_collection(tf.GraphKeys.ACTIVATIONS)
            #    for v in mv:
            #        print(v.name)
            #        print(v.outputs.get_shape().as_list()[1:])

            if ix % 10 == 0:
                print(_conf_accum)# / np.sum(_conf_accum))

    coord.request_stop()
    coord.join(threads)

