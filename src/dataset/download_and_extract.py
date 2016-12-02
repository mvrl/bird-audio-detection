import os
import zipfile
import shutil
import requests
from collections import defaultdict

label_base = 'https://ndownloader.figshare.com/files/'
audio_base = 'https://archive.org/download/'
data_base  = '../../data/'

datasets = {
    'freefield1010' : { 'labels' : '6035814',
                        'audio'  : 'ff1010bird/ff1010bird_wav.zip' },
    'warblr'        : { 'labels' : '6035817',
                        'audio'  : 'warblrb10k_public/warblrb10k_public_wav.zip' }
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
                
if not os.path.exists(data_base + 'ff1010_audio/'):
    os.mkdir(data_base + 'ff1010_audio/')

if not os.path.exists(data_base + 'warblr_audio/'):
    os.mkdir(data_base + 'warblr_audio/')

for key in datasets:
    print('Downloading %s labels' % key)
    get_file(label_base + datasets[key]['labels'], 
             data_base + '%s_labels.csv' % key)

    print('Downloading %s audio' % key)
    get_file(audio_base + datasets[key]['audio'], 
             data_base + '%s_audio.zip' % key)

    print('Unzipping %s audio' % key)
    unzip_file(data_base + '%s_audio.zip' % key, data_base + '%s_audio/' % key)
