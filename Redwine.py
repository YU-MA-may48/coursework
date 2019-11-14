#!/usr/bin/env python
# coding:utf-8

import numpy as np
#import np.array
import nltk
import sklearn
import operator
import requests
from nltk.corpus import stopwords
from sklearn.svm import SVC

nltk.download('stopwords') # If needed
nltk.download('punkt') # If needed
nltk.download('wordnet') # If needed

#path= '/Users/mayu/OneDrive - Cardiff University/applied machine learning/Coursework/datasets_coursework1/Wine'
path= '/Users/mayu/OneDrive - Cardiff University/applied machine learning/Coursework/datasets_coursework1/Wine/wine_test.csv'
dataset_file=open(path).readlines()

count_line=0
for wine_line in dataset_file:
 count_line +=1
for wine_line in dataset_file[1:5]:
    print ("Redwine_line" +str(count_line)+ ":"+str(wine_line))

# i = 1
# for wine_line in dataset_file[:3]:
#     i += 1
#     print ("Redwine_line : " + str(wine_line))

X_train=[]
Y_train=[]
for wine_line in dataset_file[1:]:
  wine_linesplit=wine_line.split(";")
  vector_wine_features=np.zeros(len(wine_linesplit)-1)
  for i in range(len(wine_linesplit)-1):
    vector_wine_features[i]=float(wine_linesplit[i])
  X_train.append(vector_wine_features)
  Y_train.append(int(wine_linesplit[-1]))

X_train_diabetes=np.asarray(X_train)
Y_train_diabetes=np.asarray(Y_train)
svm_clf_diabetes=sklearn.svm.SVC(gamma='auto') # Initialize the SVM model
svm_clf_diabetes.fit(X_train_diabetes,Y_train_diabetes) # Train the SVM model

wine_1=['7.2', '0.23', '0.32', '8.5', '0.058', '47', '186', '0.9956' , '3.19' , '0.4', '9.9']
wine_2=['5.4', '5.55', '0.4', '3.2', '0.05', '80', '30', '0.9951', '3.26', '0.44', '6.8']
print (svm_clf_diabetes.predict([wine_1]))
print (svm_clf_diabetes.predict([wine_2]))


X_train=[]
Y_train=[]
selected_features = [1,3,5]
for wine_line in dataset_file:
#    print("wine line: " + wine_line)
    wine_linesplit = wine_line.split(",")
    vector_wine_features = np.zeros(len(selected_features))
    feature_index = 0
    for i in range(len(wine_linesplit) - 1):
        if i in selected_features:
            vector_wine_features[feature_index] = float(wine_linesplit[i])
            feature_index += 1
#            print("vec wine feat" + str(vector_wine_features))
            X_train.append(vector_wine_features)
            Y_train.append(int(wine_linesplit[-1]))

#svm_clf = sklearn.svm.SVC(kernel="linear", gamma='auto')  # Load the (linear) SVM model
#svm_clf.fit(X_train, Y_train)  # Train the SVM model
#svm_clf.fit(X_train)  # Train the SVM model

#print("x train data: " + str(X_train))

X_train_diabetes=np.asarray(X_train)
Y_train_diabetes=np.asarray(Y_train) # This step is really not necessary, but it is recommended to work with numpy arrays instead of Python lists.

svm_clf_diabetes=sklearn.svm.SVC(kernel="linear",gamma='auto') # Initialize the SVM model
#svm_clf_diabetes=sklearn.svm.SVC(gamma='auto') # Initialize the SVM model
#print("x train diabetes data: " + str(X_train_diabetes))

svm_clf_diabetes.fit(X_train_diabetes,Y_train_diabetes) # Train the SVM model

wine_1 = ['0.23', '8.5', '47']
wine_2 = ['0.28', '6.9', '30']
print (svm_clf.predict([wine_1]))
print (svm_clf.predict([wine_2]))






