import tensorflow as tf
import dataset

feat, label, recname = dataset.records_challenge()

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    
    # This is necessary for initializing num_epoch local variable
    sess.run(tf.local_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        while(True):
            _feat,_label,_recname = sess.run([feat,label,recname])

            print(len(_label))

    except tf.errors.OutOfRangeError:
        print('Queue empty, exiting now...')
        coord.request_stop()
        coord.join(threads)
