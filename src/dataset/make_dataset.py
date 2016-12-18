from sklearn.model_selection import KFold

DATA_BASE = '../../data/'
datasets = [ 'freefield1010', 'warblr', 'badchallenge' ]
test_size = 0.1
random_state = 0
num_folds = 10

def split_dataset(dataset_name):
    f_lines = []
    with open(DATA_BASE + dataset_name + '_labels.csv', 'r') as fb:
        fb.readline()
        for line in fb:
            # Account for test set, which we need a dummy label
            if line.split(',')[1] == '\n':
                line = line[:-1] + '-1\n'
            f_lines.append(dataset_name + '_audio/wav/' + line)

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    counter = 0
    for _, fold_index in kf.split(f_lines):
        with open(dataset_name + '_%02d.csv' % counter, 'w') as fb:
            for index in fold_index:
                fb.write(f_lines[index])
        counter += 1

# Assuming labels are downloaded at DATA_BASE location
for dataset_name in datasets:
    split_dataset(dataset_name)
