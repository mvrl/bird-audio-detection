'''
Purpose:  this code test the provide trained model on a set of test files.
Input: the trained model, and the the test files and their features. 
Output: the ROC curve with the AUC value on the figure. 

Tawfiq Salem
'''

import numpy as np
import h5py
from sklearn import svm
from sklearn import datasets
import glob
from sklearn import metrics
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import random
import sklearn.decomposition
import os

features_dir="/u/eag-d1/scratch/jacobs/birddetection/"
folds="/u/eag-d1/scratch/jacobs/birddetection/folds/"

def get_test_files():
    #get the paths for the testing files
    test_file_f9=open(folds+"freefield1010_09.csv","r")
    test_file_w9=open(folds+"warblr_09.csv","r")
  

    lines_f9=test_file_f9.readlines()
    lines_w9=test_file_w9.readlines()
    lines=lines_f9+lines_w9  
    return lines


def load_data(files_path,target_features, layer_name):
    #Read the h5 files(soundnet features h5) 
    training_set=[]
    training_labels=[]
    for indx,line in enumerate(files_path):
        print indx
        label=line.split("/")[-1].split(",")[-1].replace("\n","")
        sound_id=line.split("/")[-1].split(",")[-2].replace("\n","")
        dataset_name=line.split("/")[-3]
        h5_file=features_dir+dataset_name+"/features/"+target_features+"/"+sound_id+".wav.soundnet.h5"
        hf=h5py.File(h5_file,'r')
        data=hf.get(layer_name)
        np_data=np.array(data)
        np_data=np_data.sum(axis=1)
        #flat_data=np_data.ravel()
        #print flat_data.shape
        training_set.append(np_data)
        training_labels.append(int(label))
 
    train_data=np.float32(training_set)
    #train_data=np.float32(train_data)
    print train_data.shape
    print "The shape of the training data="+str(train_data.shape)
    train_labels=np.int_(training_labels)
    return train_data,train_labels

def main():
    target_features="soundnet_pool5"
    layer_name="layer18"

    test_lines= get_test_files()

    test_data,test_labels=load_data(test_lines,target_features,layer_name)
    
    svm_linear_clf= joblib.load('./svm_linear_pool5_model.pkl') 
    pred_scores_svm=svm_linear_clf.predict_proba(test_data)
     
    pred_scores_svm=pred_scores_svm[:,1]
    
    fpr, tpr, thresholds=metrics.roc_curve(test_labels,pred_scores_svm, pos_label=1)
    
    roc_auc=metrics.auc(fpr,tpr)
    
    lw = 2
    plt.figure()
    plt.plot(fpr_rbf, tpr_rbf, color='darkorange',lw=lw, label='SVM_rbf curve (area = %0.2f)' % roc_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC curve for different classifiers for birddetection")
    plt.legend(loc="lower right")
    plt.savefig("figs/svm_linear_sum_"+target_features+".pdf")
    plt.show()

main()
