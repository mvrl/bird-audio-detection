import tensorflow as tf
import dataset
from datetime import datetime


#feat, label, recname = dataset.records_challenge()
#feat, label, recname = dataset.records_train_fold()
#feat, label, recname = dataset.records_test_fold(dataset_names=["free"])
feat, label, recname = dataset.records_test_fold()

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    
    # This is necessary for initializing num_epoch local variable
    sess.run(tf.local_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        while(True):
            tstart = datetime.now()
            _feat,_label,_recname = sess.run([feat,label,recname])
            tend = datetime.now()
            print(tend-tstart)

            print(_recname)
            print(_label)
            print("Number of entries: %i"%len(_label))
            print("Average label: %f"%_label.mean())

    except tf.errors.OutOfRangeError:
        print('Queue empty, exiting now...')
    finally:
        coord.request_stop()
        coord.join(threads)
