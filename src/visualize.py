import os
import wave
import dataset
import network
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.misc import imresize
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import spectrogram
from scipy.cluster.hierarchy import dendrogram, linkage

slim = tf.contrib.slim

ckpt_location = '/u/eag-d1/scratch/jacobs/birddetection/checkpoints/v2.1_elu_0.20_1.00_no'

with tf.variable_scope('Input'):
    print('Defining input pipeline')
    feat, label, recname = dataset.records_test_fold()

with tf.variable_scope('Predictor'):
    print('Defining prediction network')

    logits, end_points = network.network(feat,
                                         is_training=False,
                                         activation_fn=tf.nn.elu,
                                         capacity=0.2,
                                         capacity2=1.0)

    probs = tf.nn.softmax(logits)
    prediction = tf.cast(tf.argmax(logits,1),dtype=tf.int32)

if not os.path.exists('./conv5'):
    os.mkdir('./conv5')

if not os.path.exists('./conv6'):
    os.mkdir('./conv6')

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(ckpt_location)
    if ckpt and ckpt.model_checkpoint_path: 
        print('Restoring checkpoint')
        saver.restore(sess, ckpt.model_checkpoint_path)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    _feat, _label, _recname, _prediction, output = sess.run([feat,
                                                             label, 
                                                             recname, 
                                                             prediction, 
                                                             end_points])
    for idx in range(20):
        sample = output['conv6'][idx]

        plt.figure(1)
        plt.subplot(311)
        plt.plot(range(sample.shape[0]), sample[:,0,0])
        plt.axis((0, sample.shape[0], -25.0, 25.0))

        plt.title('%s %d %d' % (_recname[idx], _label[idx], _prediction[idx]))

        plt.subplot(312)
        plt.plot(range(sample.shape[0]), sample[:,0,1])
        plt.axis((0, sample.shape[0], -25.0, 25.0))

        plt.subplot(313)
        plt.specgram(_feat[idx])

        plt.savefig('./conv6/test_%d.png' % idx)
        plt.clf()

        sample = output['conv5'][idx]

        sample = sample.reshape((sample.shape[0], 26)).T
        sample = imresize(sample, (26,150))
        sample = gaussian_filter(sample, 1)

        # Sort sample by column 45
        # sample = [ sample[idx] for idx in np.argsort(sample[:,45]) ]

        # Sort sample by mean of each row
        mean = [ np.mean(x) for x in sample ]
        sample = [ sample[idx] for idx in np.argsort(mean) ]

        plt.figure(1)
        plt.subplot(211)
        plt.imshow(sample)
        
        plt.title('%s %d %d' % (_recname[idx], _label[idx], _prediction[idx]))

        plt.subplot(212)
        plt.specgram(_feat[idx])

        plt.savefig('./conv5/test_%d.png' % idx)
        plt.clf()
