from sklearn.model_selection import train_test_split

DATA_BASE = '../../data/'
datasets = [ 'freefield1010', 'warblr' ]
test_size = 0.1
random_state = 0
num_folds = 10

def split_dataset(dataset_name):
    f_lines = []
    with open(DATA_BASE + dataset_name + '_labels.csv', 'r') as fb:
        for line in fb:
            f_lines.append(dataset_name + '_audio/wav/' + line)

    # Assuming csv file has header, so we ignore first line of f_lines
    train_split, test_split, _, _ = train_test_split(f_lines[1:],
                                                range(0, len(f_lines)-1),
                                                test_size=test_size,
                                                random_state=random_state)

    counter = 0
    chunk_size = len(train_split) / (num_folds-1)
    for i in range(0, len(train_split), chunk_size):
        with open(dataset_name + '_0%d.csv' % counter, 'w') as fb:
            for line in train_split[i : i+chunk_size]:
                fb.write(line)
        counter += 1

    with open(dataset_name + '_09.csv', 'w') as fb:
        for line in test_split:
            fb.write(line)

# Assuming labels are downloaded at DATA_BASE location
for dataset_name in datasets:
    split_dataset(dataset_name)
