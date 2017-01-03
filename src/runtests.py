from __future__ import division, print_function, absolute_import

import numpy as np 
import tensorflow as tf
import os
import sys
import dataset

def ensure_no_dupes():

    _,_,l_train = dataset.records(is_training=False,batch_size=256,augment_add=False)
    _,_,l_test = dataset.records(is_training=True,batch_size=256,augment_add=False)

    with tf.Session() as sess:

        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print('running')

        _l_trains = []
        _l_tests = []

        for ix in xrange(1000):

            _l_train, _l_test = sess.run([l_train,l_test])

            _l_trains.extend(_l_train)
            _l_tests.extend(_l_test)

            inte = set(_l_trains) & set(_l_tests)
            if 0 != len(inte):
                assert "overlaps!!!" 
            else:
                print('No overlaps in first %d minibatches' % ix )

        coord.request_stop()
        coord.join(threads)

def main(argv):

    ensure_no_dupes()


if __name__ == "__main__":
    main(sys.argv)

