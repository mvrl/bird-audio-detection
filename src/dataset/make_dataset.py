from sklearn.model_selection import train_test_split

DATA_BASE = '../../data/'

# Assuming labels are downloaded at DATA_BASE location
with open(DATA_BASE + 'freefield1010_labels.csv', 'r') as fb:
    ff1010_lines = fb.readlines()

with open(DATA_BASE + 'warblr_labels.csv', 'r') as fb:
    warblr_lines = fb.readlines()

ff1010_train, ff1010_test, _, _ = train_test_split(ff1010_lines[1:],
                                               range(0, len(ff1010_lines)-1),
                                               test_size = 0.1,
                                               random_state=0)

warblr_train, warblr_test, _, _ = train_test_split(warblr_lines[1:],
                                               range(0, len(warblr_lines)-1),
                                               test_size = 0.1,
                                               random_state=0)

def write_csv(data, f_name):
    with open(f_name, 'w') as fb:
        for line in data:
            fb.write(line)

write_csv(ff1010_train, './ff1010_train.csv')
write_csv(ff1010_test, './ff1010_test.csv')
write_csv(warblr_train, './warblr_train.csv')
write_csv(warblr_test, './warblr_test.csv')
