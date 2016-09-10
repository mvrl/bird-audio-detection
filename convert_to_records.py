from __future__ import division, print_function, absolute_import

import os
import itertools
from multiprocessing import Pool
import traceback

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
        return

    try:
        print(dat + ' processing excel')

        # merge label files
        book = xlrd.open_workbook(datadir + dat + '.xls')
        sh = book.sheet_by_index(0)
        labels1 = np.asarray([tmp.value for tmp in
                sh.col_slice(0,start_rowx=0,end_rowx=21600)])
        book= xlrd.open_workbook(datadir + dat + '_2.xls')
        sh = book.sheet_by_index(0)
        labels2 = np.asarray([tmp.value for tmp in
                sh.col_slice(0,start_rowx=0,end_rowx=21600)])

        assert 21600 == labels1.size, dat + ' wrong number of labels (' + str(labels1.size) + ')'
        assert 21600 == labels2.size, dat + ' wrong number of labels (' + str(labels2.size) + ')'

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
            buf = buf.reshape((21600,-1))
            features[label] = buf

        print(dat + ' exporting to TFRecord format')
    except ValueError:
        traceback.print_exc()
        return

    util.export_to_records(outfile,features,labels1,labels2)

pool = Pool(processes=8)

pool.map_async(
        dump,
        [tmp.strip() for tmp in itertools.chain(
            open('train.txt','r'),open('test.txt','r'))]
        ).get(999999999)

