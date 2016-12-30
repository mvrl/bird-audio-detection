"""
    This script helps to visualize the learned filters of conv1
    and various layers' activations for network v5
"""

import os
import util
import dataset
import network
import shutil
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.misc import imresize
from scipy.ndimage.filters import gaussian_filter

slim = tf.contrib.slim

data_base = '../data/'

nc,dc = util.parse_arguments()
run_name = util.run_name(nc,dc)

checkpoint_dir = 'checkpoint/' + run_name + '/'

# Plot everything into multiple columns, so it's most squared
def plot_images(plt_title, activations, subplot_titles=True,
                series_axis=True, series_range=(-2.0, 2.0)):
    plt.close('all')
    fig = plt.figure(1)
    num_subplots = len(activations)
    num_cols = np.floor(np.sqrt(num_subplots))
    num_rows = np.ceil(num_subplots / num_cols)
    keys = activations.keys()
    keys.sort()
    counter = 1
    for key in keys:
        activ_map = activations[key]
        ax = plt.subplot(num_rows, num_cols, counter)
        if subplot_titles:
            ax.set_title('%s %s' % (key, activ_map.shape))

        # If it's a 1D array
        if activ_map.shape[0] == 1:
            activ_map = activ_map.reshape((activ_map.shape[-1],))
            plt.plot(range(activ_map.size), activ_map)
            x1, x2, (y1, y2) = 0, activ_map.size, series_range
            plt.axis((x1, x2, y1, y2))

            if not series_axis:
                plt.axis('off')
        
        # If it's a 2D array
        else:
            map_shape = (50, 100)
            image = imresize(activations[key], map_shape)
            image = gaussian_filter(image, 1)
            plt.axis('off')
            plt.imshow(image)

        counter += 1

    fig.suptitle(plt_title, size=16)
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)

with tf.variable_scope('Input'):
    print('Defining input pipeline')
    feat, label, recname = dataset.records_test_fold(**dc)

with tf.variable_scope('Predictor'):
    print('Defining prediction network')

    logits = network.network(feat, is_training=False, **nc)

    probs = tf.nn.softmax(logits)
    prediction = tf.cast(tf.argmax(logits,1),dtype=tf.int32)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    save_dir = './visualizations/%s/' % run_name
    if not os.path.exists(save_dir):
        save_dir = os.mkdir(save_dir)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path: 
        print('Restoring checkpoint')
        saver.restore(sess, ckpt.model_checkpoint_path)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Get all network end_points
    end_points = tf.get_collection(tf.GraphKeys.ACTIVATIONS, scope='Predictor')

    _feat, _label, _recname, _prediction, _end_points = sess.run([feat,
                                                                  label, 
                                                                  recname, 
                                                                  prediction, 
                                                                  end_points])

    import ipdb; ipdb.set_trace()

    # Plot filters of conv1
    print('Plotting conv1 filters')
    filters = {}
    plt_title = '%s Conv1 Filters' % run_name
    model_variables = slim.get_model_variables()

    for var in model_variables:
        if 'Conv/weights' in var.op.name:
            conv1_filters = var

    conv1_kernels = sess.run(conv1_filters)
    conv1_kernels = conv1_kernels.reshape(-1, conv1_kernels.shape[-1]).T 

    for idx, kernel in enumerate(conv1_kernels):
        filters[str(idx)] = kernel.reshape((1, kernel.size))

    plot_images(plt_title, filters, subplot_titles=False, series_axis=False)

    plt.savefig('%s/%s.png' % (save_dir,'conv1_filters'),bbox_inches='tight')
    
    for idx in range(len(_recname)):
        # Plot activations
        print('Plotting %s activations' % _recname[idx])
        activations = {}
        plt_title = '%s,%d,%d' % (_recname[idx], _label[idx], _prediction[idx])
        keys = _end_points.keys() 
        keys.sort()
        for key in keys:
            activ_map = _end_points[key][idx]
            activ_map = activ_map.reshape(-1, activ_map.shape[-1])
            
            if activ_map.shape[0] > activ_map.shape[1]:
                activ_map = activ_map.T
            
            activations[key] = activ_map

        plot_images(plt_title, activations)
        
        shutil.copy(data_base + _recname[idx] + '.wav', 
                    '%s/%d.wav' % (save_dir, idx))

        plt.savefig('%s/%d.png' % (save_dir, idx), bbox_inches='tight')
