from __future__ import division, print_function, absolute_import

import os
from optparse import OptionParser
import tensorflow as tf
import ops

def parse_arguments():

    parser = OptionParser()

    parser.add_option("-a", "--activation", dest="AFN", default='relu')
    parser.add_option("-n", "--network", dest="NET", default='v2')
    #parser.add_option("-f", "--feature", dest="FEAT", default='eeg')
    parser.add_option("-c", "--capacity", dest="CAP", type="float", default=1.0)

    (opts, args) = parser.parse_args()

    nc = {}
    nc['activation_fn'] = {
        'elu':tf.nn.elu,
        'relu':tf.nn.relu,
        'lrelu':ops.lrelu
        }[opts.AFN]

    nc['network'] = opts.NET

    nc['capacity'] = opts.CAP

    return nc

def run_name(nc):

    # define run name
    run_name = nc['network']
    run_name += '_' + nc['activation_fn'].func_name

    run_name += '_{:0.2f}'.format(nc['capacity'])

    return run_name

