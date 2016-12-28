'''
Purpose: this code generates predictions for the test set of sound files provided by bird audio detection challenge.

Input: mainly, the trained model, and the test file.

Output: a csv file that can be submit to the
        bird audio detection challenge server.

Note: before run, you need to specify the correct paths. 

Tawfiq Salem

'''

import numpy as np
import h5py
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn import datasets
import glob
from sklearn import metrics
from sklearn.externals import joblib
import matplotlib.pyplot as plt

features_dir="/u/eag-d1/scratch/jacobs/birddetection/badchallenge_audio/features/"
output_file=open ("results_test_v1.csv","w")
output_file.write("itemid,hasbird\n")
clf = joblib.load('/u/amo-d0/grad/salem/projects/birddetection/svm_pool5/models/pool5/svm_linear/svm_linear_pool5_model.pkl') 
target_features="soundnet_pool5"
layer_name="layer18"
test_file=open("/u/amo-d0/grad/salem/projects/birddetection/badch_testset_blankresults.csv","r")
lines=test_file.readlines()[1:]

for idx,line in enumerate(lines):
  print idx
  file_name=line.split(",")[0]         
  h5_file=features_dir+target_features+"/"+file_name+".wav.soundnet.h5"
  hf=h5py.File(h5_file,'r')
  data=hf.get(layer_name)
  np_data=np.array(data)
  np_data=np_data.sum(axis=1)
  np_data=np_data.reshape(1,-1)
  pred_scores_svm=clf.predict_proba(np_data)
  output_file.write(file_name+","+str(pred_scores_svm[0][1])+"\n")
  print pred_scores_svm[0][1]


output_file.close()   
   
