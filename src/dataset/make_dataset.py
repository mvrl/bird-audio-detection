from sklearn.model_selection import train_test_split

DATA_BASE = '../../data/'
datasets = [ 'freefield1010', 'warblr' ]
test_size = 0.1
random_state = 0

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

    with open(dataset_name + '_train.csv', 'w') as fb:
        for line in train_split:
            fb.write(line)

    with open(dataset_name + '_test.csv', 'w') as fb:
        for line in test_split:
            fb.write(line)

# Assuming labels are downloaded at DATA_BASE location
for dataset_name in datasets:
    split_dataset(dataset_name)
