import tensorflow as tf
import dataset

feat, label, recname = dataset.records(batch_size=20)

with tf.Session() as sess:

    sess.run(tf.initialize_all_variables())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        for ix in xrange(100):
            _feat,_label,_recname = sess.run([feat,label,recname])

            print(_label)
            print(_label.mean())
            #print(_recname)
    finally:
        coord.request_stop()
        coord.join(threads)

