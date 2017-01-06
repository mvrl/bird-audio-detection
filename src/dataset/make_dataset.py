from __future__ import division, print_function, absolute_import

from sklearn.model_selection import KFold
from sklearn.utils import resample
import numpy as np
import itertools
import random 
from collections import defaultdict

DATA_BASE = '../../data/'
datasets_train = [ 'freefield1010', 'warblr']
datasets_challenge = ['badchallenge' ]
random_state = 0
num_folds = 10

negative_samples = []

def split_dataset(dataset_name, balance=False, shuffle=True):

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
            names.append(dataset_name + '_audio/wav_22050/' + name)
            labels.append(label)


    print('Splitting into folds')
    kf = KFold(n_splits=num_folds, shuffle=shuffle, random_state=random_state)

    for counter, (_, fold_index) in enumerate(kf.split(names)):

        samples = defaultdict(list) 

        # split into different classes
        for index in fold_index:
            samples[labels[index]].append((names[index],labels[index]))

        fold_filename = dataset_name + '_%02d.csv' % counter
        print('Exporting %s' % fold_filename)

        negative_samples.extend(samples['0'])

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
        if shuffle:
            random.shuffle(items)

        # output items
        with open(fold_filename, 'w') as fid:
            for item in items:
                print("%s,%s" % item, file=fid)

# Assuming labels are downloaded at DATA_BASE location
for dataset_name in datasets_train:
    split_dataset(dataset_name, balance=True, shuffle=True)

for dataset_name in datasets_challenge:
    split_dataset(dataset_name, balance=False, shuffle=False)

# Shuffle and save all negative samples to a file
random.shuffle(negative_samples)
print('Writing negative samples')
with open('./negative_samples.csv', 'w') as fb:
    for sample in negative_samples:
        fb.write('%s,%s\n' % sample)
