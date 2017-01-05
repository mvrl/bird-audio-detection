from __future__ import division, print_function, absolute_import

import os
from optparse import OptionParser
import tensorflow as tf
import ops

def parse_arguments():

    parser = OptionParser()

    parser.add_option("-a", "--activation", dest="AFN", default='relu')
    parser.add_option("-n", "--network", dest="NET", default='v2')
    parser.add_option("-c", "--capacity", dest="CAP", type="float", default=1.0)
    parser.add_option("-b", "--capacity2", dest="CAP2", type="float", default=1.0)
    parser.add_option("-A", action="store_true", dest="AUG", default=False)
    parser.add_option("-m", "--message", action="store", type="string", dest="msg", default="")

    (opts, args) = parser.parse_args()

    # specify network configuration
    nc = {}
    nc['activation_fn'] = {
        'elu':tf.nn.elu,
        'relu':tf.nn.relu,
        'lrelu':ops.lrelu
        }[opts.AFN]

    nc['network'] = opts.NET

    nc['capacity'] = opts.CAP
    nc['capacity2'] = opts.CAP2

    # specify dataset configuration
    dc = {}
    dc['augment_add'] = opts.AUG 

    # specify run configuration
    rc = {}
    rc['message'] = opts.msg

    return nc, dc, rc

def run_name(nc,dc,rc):

    # define run name
    run_name = nc['network']
    run_name += '_' + nc['activation_fn'].func_name

    run_name += '_{:0.2f}'.format(nc['capacity'])
    run_name += '_{:0.2f}'.format(nc['capacity2'])

    run_name += '_yes' if dc['augment_add'] else '_no'

    run_name += '_' + rc['message'] if rc['message'] != '' else ''

    return run_name

