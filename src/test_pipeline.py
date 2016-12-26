import tensorflow as tf
import dataset
from datetime import datetime


tensor_list = {
        'challenge_add':dataset.records_challenge(augment_add=True), 
        'challenge':dataset.records_challenge(), 
        'train_fold':dataset.records_train_fold(),
        'test_fold_free':dataset.records_test_fold(dataset_names=["free"]),
        'test_fold':dataset.records_test_fold()}

for name,(feat, label, recname) in tensor_list.items(): 
    print("======================================")
    print(name)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        
        # This is necessary for initializing num_epoch local variable
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            for ix in range(10):
                tstart = datetime.now()
                _feat,_label,_recname = sess.run([feat,label,recname])
                tend = datetime.now()
                print(tend-tstart)

                print(_recname[0:10])
                print(_label[0:10])
                print("Number of entries: %i"%len(_label))
                print("Average label: %f"%_label.mean())

        except tf.errors.OutOfRangeError:
            print('Queue empty, exiting now...')
        finally:
            coord.request_stop()
            coord.join(threads)
