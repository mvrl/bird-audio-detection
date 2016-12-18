import os
import zipfile
import shutil
import requests

data_base  = '../../data/'

datasets = {
    'freefield1010' : { 'labels' : 'https://ndownloader.figshare.com/files/6035814',
                        'audio'  : 'https://archive.org/download/ff1010bird/ff1010bird_wav.zip' },
    'warblr'        : { 'labels' : 'https://ndownloader.figshare.com/files/6035817',
                        'audio'  : 'https://archive.org/download/warblrb10k_public/warblrb10k_public_wav.zip' },
    'badchallenge'  : { 'labels' : 'https://archive.org/download/birdaudiodetectionchallenge_test/badch_testset_blankresults.csv',
                        'audio'  : 'https://archive.org/download/birdaudiodetectionchallenge_test/badchallenge_testdata_wavs.zip' }
}

def unzip_file(source, dest):
    zip_ref = zipfile.ZipFile(source, 'r')
    zip_ref.extractall(dest)
    zip_ref.close()

def get_file(url, f_name):
    with open(f_name, 'wb') as f:
        response = requests.get(url, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None:
            f.write(response.content)
        else:
            downloaded = 0.0
            previous = -1.0
            total_length = float(total_length)
            for data in response.iter_content(chunk_size=4096):
                downloaded += len(data)
                f.write(data)
                done = int(downloaded / total_length * 100)
                if done != previous:
                    print '%d%% Downloaded!' % done
                    previous = done

if not os.path.exists(data_base):
    os.mkdir(data_base)

for key in datasets:
    # Check if label file is downloaded
    if os.path.exists(data_base + '%s_labels.csv' % key):
        print(
            '%s labels exist already, delete %s_labels.csv if wish to redownload.' % (key, key))
    else:
        print('Downloading %s labels' % key)
        get_file(datasets[key]['labels'], data_base + '%s_labels.csv' % key)

    # Check if audio files are downloaded
    if os.path.exists(data_base + '%s_audio/' % key):
        print('%s audio files exists already, delete %s_audio.zip and %s_audio/ if wish to redownload.' % (key, key, key))

    else:
        os.mkdir(data_base + '%s_audio' % key)

        print('Downloading %s audio' % key)
        get_file(datasets[key]['audio'], data_base + '%s_audio.zip' % key)

        print('Unzipping %s audio' % key)
        unzip_file(data_base + '%s_audio.zip' % key, 
                   data_base + '%s_audio/' % key)

print('Download complete, delete .zip files if you would like to save space.') 
