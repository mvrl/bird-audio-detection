from __future__ import division, print_function, absolute_import

import os
import itertools
from multiprocessing import Pool

import util

import numpy as np
import pyedflib
import xlrd
    
datadir = './data/'
recorddir = './records/'

def dump(dat):

    outfile = recorddir + dat + '.tfrecord'

    if os.path.isfile(outfile):
        print('skipping ' + outfile)

    try:
        print(dat + ' processing excel')

        # merge label files
        book = xlrd.open_workbook(datadir + dat + '.xls')
        sh = book.sheet_by_index(0)
        labels = np.asarray([tmp.value for tmp in sh.col_slice(0)])
        book= xlrd.open_workbook(datadir + dat + '_2.xls')
        sh = book.sheet_by_index(0)
        labels = np.concatenate((
            labels,
            np.asarray([tmp.value for tmp in sh.col_slice(0)])))

        print(dat + ' processing edf')
        f = pyedflib.EdfReader(datadir + dat + '.edf')
        features = {} 
        for channel in range(4):
            label = f.getLabel(channel)
            print(dat + ' reading ' + label )
            if label.startswith('P'):
                print(dat + ' renaming ' + label + ' to ' + 'Piezo')
                label = 'Piezo'
            
            buf = f.readSignal(channel)
            buf = buf.reshape((labels.size,-1))
            features[label] = buf

        print(dat + ' exporting to TFRecord format')
        util.export_to_records(outfile, features,labels)
    except ValueError:
        traceback.print_exc()
        print('removing ' + recorddir + dat + '.tfrecord')
        os.remove(recorddir + dat + '.tfrecord')

pool = Pool(processes=7)

pool.map_async(
        dump,
        [tmp.strip() for tmp in itertools.chain(
            open('train.txt','r'),open('test.txt','r'))]
        ).get(999999999)

