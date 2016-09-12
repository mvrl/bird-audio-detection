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

    # weights to make sure infrequent classes have higher weight 
    # TODO make a moving average of the weight 
    _, idx, count = tf.unique_with_counts(label)
    count = tf.cast(count,tf.float32)
    weight = 3.*tf.gather(tf.reduce_mean(count) / (1. + count),idx)

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
    acc_match = tf.contrib.metrics.accuracy(label1,label2)
    conf = tf.contrib.metrics.confusion_matrix(prediction,label,num_classes=tf.cast(3,tf.int64),dtype=tf.int64)

with tf.Session() as sess:

    update_ops = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    sess.run(tf.initialize_all_variables())

    for ix in xrange(70000):
        #print(sess.run((tf.reduce_mean(features),tf.reduce_mean(tf.square(features)))))
        #continue
        _,_,_i,_loss,_acc,_acc_match,_conf = sess.run([
            train_op,
            update_ops,
            global_step,
            loss,
            acc,
            acc_match,
            conf])
        print(str(_i) + ' : ' + str(_loss) + ' : ' + str(_acc) + ' : ' + str(_acc_match))

        if ix % 10 == 0:
            print(_conf)
        
        #print(_y)
        #print(_count)

    coord.request_stop()
    coord.join(threads)

