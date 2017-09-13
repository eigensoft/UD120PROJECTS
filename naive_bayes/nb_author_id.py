#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
import os
from time import time
from sklearn.naive_bayes import GaussianNB
current_dir = os.getcwd() # find the current directory
sys.path.append("tools\\")
from email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels




features_train, features_test, labels_train, labels_test = preprocess()




#########################################################

clf =GaussianNB()


clf.fit(features_train,labels_train)

accuracy2=clf.score(features_test,labels_test,sample_weight=None)


#########################################################


