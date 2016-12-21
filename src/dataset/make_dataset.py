from __future__ import division, print_function, absolute_import

from sklearn.model_selection import KFold
from sklearn.utils import resample
import numpy as np
import itertools
from random import shuffle
from collections import defaultdict

DATA_BASE = '../../data/'
datasets_train = [ 'freefield1010', 'warblr']
datasets_challenge = ['badchallenge' ]
random_state = 0
num_folds = 10

def split_dataset(dataset_name, balance=False):

    print('Spliting %s.' % dataset_name)

    names = []
    labels = []
    with open(DATA_BASE + dataset_name + '_labels.csv', 'r') as fb:
        fb.readline()
        for line in fb:
            name, label = line.split(',')
            label = label.strip()
            if label == '':
                label = '-1'
            names.append(dataset_name + '_audio/wav/' + name)
            labels.append(label)


    print('Splitting into folds')
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)

    for counter, (_, fold_index) in enumerate(kf.split(names)):

        samples = defaultdict(list) 

        # split into different classes
        for index in fold_index:
            samples[labels[index]].append((names[index],labels[index]))

        fold_filename = dataset_name + '_%02d.csv' % counter
        print('Exporting %s' % fold_filename)

        if balance:
            print('Balancing each fold')
            n_pos = len(samples['1'])
            n_neg = len(samples['0'])
            if n_neg < n_pos:
                tmp = resample(samples['0'],n_samples=n_pos-n_neg)
                samples['0'].extend(tmp)
            elif n_pos < n_neg:
                tmp = resample(samples['1'],n_samples=n_neg-n_pos)
                samples['1'].extend(tmp)

        items = list(itertools.chain(*samples.values()))
        shuffle(items)

        # output items
        with open(fold_filename, 'w') as fid:
            for item in items:
                print("%s,%s" % item, file=fid)

# Assuming labels are downloaded at DATA_BASE location
for dataset_name in datasets_train:
    split_dataset(dataset_name, balance=True)

for dataset_name in datasets_challenge:
    split_dataset(dataset_name, balance=False)

