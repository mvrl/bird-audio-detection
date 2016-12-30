'''
Purpose: this program is to train svm for classify birds' sound audio files
Input: the paths of the train files and the labels, the paths for the features of the train files
Output: the trained model. 

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import random
import sklearn.decomposition
import os
import glob

features_dir="/u/eag-d1/scratch/jacobs/birddetection/"
folds="/u/eag-d1/scratch/jacobs/birddetection/folds/"


def get_train_files():
    #get the paths for the training files.
    #TODO: clean this function and do it in a more efficient way   
    data_file_f0=open(folds+"freefield1010_00.csv","r")
    data_file_f1=open(folds+"freefield1010_01.csv","r")
    data_file_f2=open(folds+"freefield1010_02.csv","r")
    data_file_f3=open(folds+"freefield1010_03.csv","r")
    data_file_f4=open(folds+"freefield1010_04.csv","r")
    data_file_f5=open(folds+"freefield1010_05.csv","r")
    data_file_f6=open(folds+"freefield1010_06.csv","r")
    data_file_f7=open(folds+"freefield1010_07.csv","r")
    data_file_f8=open(folds+"freefield1010_08.csv","r")
 
    data_file_w0=open(folds+"warblr_00.csv","r")
    data_file_w1=open(folds+"warblr_01.csv","r")
    data_file_w2=open(folds+"warblr_02.csv","r")
    data_file_w3=open(folds+"warblr_03.csv","r")
    data_file_w4=open(folds+"warblr_04.csv","r")
    data_file_w5=open(folds+"warblr_05.csv","r")
    data_file_w6=open(folds+"warblr_06.csv","r")
    data_file_w7=open(folds+"warblr_07.csv","r")
    data_file_w8=open(folds+"warblr_08.csv","r")

    lines_f0=data_file_f0.readlines()
    lines_f1=data_file_f1.readlines()
    lines_f2=data_file_f2.readlines()
    lines_f3=data_file_f3.readlines()
    lines_f4=data_file_f4.readlines()
    lines_f5=data_file_f5.readlines()
    lines_f6=data_file_f6.readlines()
    lines_f7=data_file_f7.readlines()
    lines_f8=data_file_f8.readlines()
    lines_w0=data_file_w0.readlines()
    lines_w1=data_file_w1.readlines()
    lines_w2=data_file_w2.readlines()
    lines_w3=data_file_w3.readlines()
    lines_w4=data_file_w4.readlines()
    lines_w5=data_file_w5.readlines()
    lines_w6=data_file_w6.readlines()
    lines_w7=data_file_w7.readlines()
    lines_w8=data_file_w8.readlines()
   # 

   # # lines represents all the files' path that will be use for training 
 
    lines=lines_f0+lines_f1+lines_f2+lines_f3+lines_f4+lines_f5+lines_f6+lines_f7+lines_f8+\
    lines_w0+lines_w1+lines_w2+lines_w3+lines_w4+lines_w5+lines_w6+lines_w7+lines_w8
    random.shuffle(lines)
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
    print train_data.shape
    print "The shape of the training data="+str(train_data.shape)
    train_labels=np.int_(training_labels)
    return train_data,train_labels


def train_svm_linear(train_data,train_labels):
    print "train the Linear_SVM model ....."
    C = 1.0  # SVM regularization parameter
    clf_linear = svm.SVC(kernel='linear', C=C,probability=True).fit(train_data, train_labels)
    return clf_linear 

def train_svm_rbf(train_data,train_labels):
    print "train the rbf_SVM model ....."
    C = 1.0  # SVM regularization parameter
    clf_rbf = svm.SVC(kernel='rbf', C=C,probability=True).fit(train_data, train_labels)
    return clf_rbf 


def main():
    files_path=get_train_files() 
    target_features="soundnet_pool5"
    layer_name="layer18"
    train_data,train_labels=load_data(files_path,target_features,layer_name)

    clf_linear=train_svm_linear(train_data,train_labels)

    save_dir="./pool5/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    joblib.dump(clf_linear, save_dir+"svm_linear/svm_linear_pool5_model.pkl") 


main()
