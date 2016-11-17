import tensorflow as tf
import dataset

feat, label, recname = dataset.records(
        '/u/eag-d1/scratch/jacobs/birddetection/ff1010bird_metadata.csv',
        is_training=False)

with tf.Session() as sess:

    #init_op = tf.initialize_all_variables()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    for i in range(10):
        _feat,_label,_recname = sess.run([feat,label,recname])
        print(_recname)

