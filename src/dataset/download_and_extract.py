import os
import zipfile
import shutil
import requests

freefield1010_labels = 'https://ndownloader.figshare.com/files/6035814'
freefield1010_audio  = 'https://archive.org/download/ff1010bird/ff1010bird_wav.zip'
warblr_labels = 'https://ndownloader.figshare.com/files/6035817'
warblr_audio  = 'https://archive.org/download/warblrb10k_public/warblrb10k_public_wav.zip'

def unzip_file(source, dest):
    zip_ref = zipfile.ZipFile(source, 'r')
    zip_ref.extractall(dest)
    zip_ref.close()

def get_file(url, f_name):
    with open(f_name, 'wb') as f:
        print 'Downloading %s' % f_name
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

if not os.path.exists('../../data'):
    os.mkdir('../../data')
                
if not os.path.exists('./ff1010_audio'):
    os.mkdir('../../data/ff1010_audio')

if not os.path.exists('./warblr_audio'):
    os.mkdir('../../data/warblr_audio')

print 'Downloading freefield labels'
get_file(freefield1010_labels, '../../data/ff1010bird_metadata.csv')
print 'Downloading freefield audio'
get_file(freefield1010_audio, '../../data/ff1010_audio.zip')
print 'Downloading warblr labels'
get_file(warblr_labels, '../../data/warblr_labels.csv')
print 'Downloading warblr audio'
get_file(warblr_audio, '../../data/warblr_audio.zip')

unzip_file('../../data/ff1010_audio.zip', '../../data/ff1010_audio')
unzip_file('../../data/warblr_audio.zip', '../../data/warblr_audio')
