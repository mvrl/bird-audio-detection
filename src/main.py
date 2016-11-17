# TODO non-uniform cost matrix (penalize false REM)
# TODO add wake / sleep loss 

from __future__ import division, print_function, absolute_import

import os

import tensorflow as tf
import dataset
import network
import util
slim = tf.contrib.slim

print('Setting up run')
nc = util.parse_arguments()
run_name = util.run_name(nc)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint_dir', 'checkpoint/' + run_name + '/','output directory for model checkpoints')
flags.DEFINE_string('summary_dir', 'logs/' + run_name + '/','output directory for training summaries')
flags.DEFINE_float('gamma',0.5,'learning rate change per step')
flags.DEFINE_float('learning_rate',0.03,'learning rate change per step')

if not tf.gfile.Exists(FLAGS.checkpoint_dir):
    print('Making checkpoint dir')
    os.makedirs(FLAGS.checkpoint_dir)

if not tf.gfile.Exists(FLAGS.summary_dir):
    print('Making summary dir')
    os.makedirs(FLAGS.summary_dir)

#
# Define graph 
#

with tf.variable_scope('Input'):
    print('Defining input pipeline')

    features, label, _, _ = dataset.records(
            '/u/eag-d1/scratch/jacobs/birddetection/ff1010bird_metadata.csv',
            is_training=True)

    label = tf.reshape(label,[-1])

with tf.variable_scope('Predictor'):
    print('Defining prediction network')

    logits = network.network(features,
            is_training=True,**nc)
            #use_eeg=nc['use_eeg'],
            #activation_fn=nc['activation_fn'],
            #capacity=nc['capacity'])

    # replicate because we have two annotaters
    logits = tf.concat(0,(logits,logits))

with tf.variable_scope('Loss'):
    print('Defining loss functions')

    reg = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits,
            label)

    prediction = tf.argmax(logits,1)

    loss_class = 10*tf.reduce_mean(loss_class)
    #loss_class = tf.reduce_mean(loss_class)

    loss = loss_class + reg 

with tf.variable_scope('Train'):
    print('Defining training methods')

    global_step = tf.Variable(0,name='global_step',trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,global_step,1000,FLAGS.gamma,staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate,epsilon=.1)
    train_op = optimizer.minimize(loss,global_step=global_step)

    acc = tf.contrib.metrics.accuracy(prediction,label)
    acc_match = tf.contrib.metrics.accuracy(label1,label2)
    conf = tf.contrib.metrics.confusion_matrix(prediction,label,num_classes=tf.cast(3,tf.int64),dtype=tf.int64)

with tf.Session() as sess:

    update_ops = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    sess.run(tf.initialize_all_variables())

    saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path: 
        print('Restoring checkpoint')
        saver.restore(sess, ckpt.model_checkpoint_path)

    _i = sess.run(global_step)

    print('Starting training')
    while _i < 10000:

        sess.run([features,label])

        _,_,_i,_loss,_acc,_acc_match,_conf = sess.run([
            train_op,
            update_ops,
            global_step,
            loss,
            acc,
            acc_match,
            conf
            ])
        print(str(_i) + ' : ' + str(_loss) + ' : ' + str(_acc) + ' : ' + str(_acc_match))

        if _i % 10 == 0:
            print(_conf)

	if _i % 1000 == 0:
	    print("saving total checkpoint")
	    saver.save(sess, FLAGS.checkpoint_dir + 'model.ckpt', global_step=_i)

    coord.request_stop()
    coord.join(threads)

