from __future__ import division, print_function, absolute_import

import tensorflow as tf
import dataset
import network

slim = tf.contrib.slim

with tf.variable_scope('Input'):

    class_weights = [1/250.,1/250.,1/40.]

    features, label1, label2 = dataset.records('train.txt')

    # why is this necessary?
    label1 = tf.reshape(label1,[-1])
    label2 = tf.reshape(label2,[-1])

    label = tf.concat(0,(label1,label2))

    print(label1.get_shape().as_list())
    print(label2.get_shape().as_list())
    print(label.get_shape().as_list())

    weight = 40.*tf.gather(class_weights,label)

with tf.variable_scope('Predictor'):
    logits = network.network(features)

    # replicate because we have two annotaters
    logits = tf.concat(0,(logits,logits))

with tf.variable_scope('Loss'):

    reg = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits,
            label)

    prediction = tf.argmax(logits,1)

    loss_class = 10*tf.reduce_mean(weight*loss_class)
    #loss_class = tf.reduce_mean(loss_class)

    loss = loss_class + reg 

with tf.variable_scope('Train'):
    global_step = tf.Variable(0,name='global_step',trainable=False)
    learning_rate = tf.train.exponential_decay(.1,global_step,10000,.5,staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate,epsilon=.1)
    train_op = optimizer.minimize(loss,global_step=global_step)

    acc = tf.contrib.metrics.accuracy(prediction,label)
    y, idx, count = tf.unique_with_counts(label)
    conf = tf.contrib.metrics.confusion_matrix(prediction,label,num_classes=tf.cast(3,tf.int64),dtype=tf.int64)

with tf.Session() as sess:

    update_ops = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    sess.run(tf.initialize_all_variables())

    for ix in xrange(100000):
        #print(sess.run([weight]))
        _,_,_i,_loss,_acc,_count,_y,_conf = sess.run([
            train_op,
            update_ops,
            global_step,
            loss,
            acc,
            count,y,
            conf])
        print(str(_i) + ' : ' + str(_loss) + ' : ' + str(_acc))
        if ix % 10 == 0:
            print(_conf)
        #print(_y)
        #print(_count)

    coord.request_stop()
    coord.join(threads)

