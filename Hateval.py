#!/usr/bin/env python
# coding:utf-8

import numpy as np
import nltk
import sklearn
import operator
import requests
import pandas as pd
from nltk.corpus import stopwords
from sklearn.svm import SVC

nltk.download('stopwords') # If needed
nltk.download('punkt') # If needed
nltk.download('wordnet') # If needed

path= '/Users/mayu/OneDrive - Cardiff University/applied machine learning/Coursework/datasets_coursework1/Hateval/hateval.tsv'
tsv=open(path).readlines()
tsv_read = pd.read_csv(path, sep='\t')
for row in tsv_read:
print(tsv_read.head(10))

list_tokens=nltk.tokenize.word_tokenize(paragraph)
print ("List_tokens: " +str(list_tokens))

tsv_labels=[]
tsv_data=[]
with open(path,'r') as tsv_in:
    tsv_reader = csv.reader(tsv_in, delimiter='\t')
tsv_labels = tsv_reader.__next__()
for record in tsv_reader:
    tsv_data.append(record)
    print(tsv_labels, '\n')
    print(tsv_data[0:10], '\n')


# dataset_full=[]
# for pos_review in dataset_file_pos:
#   dataset_full.append((pos_review,1))
# for neg_review in dataset_file_neg:
#   dataset_full.append((neg_review,0))

size_dataset_full=len(dataset_full)
size_test=int(round(size_dataset_full*0.2,0))

list_test_indices=random.sample(range(size_dataset_full), size_test)
train_set=[]
test_set=[]
for i,example in enumerate(dataset_full):
  if i in list_test_indices: test_set.append(example)
  else: train_set.append(example)

random.shuffle(train_set)
random.shuffle(test_set)


def get_train_test_split(dataset_full, ratio):
    train_set = []
    test_set = []
    size_dataset_full = len(dataset_full)
    size_test = int(round(size_dataset_full * ratio, 0))
    list_test_indices = random.sample(range(size_dataset_full), size_test)
    for i, example in enumerate(dataset_full):
      if i in list_test_indices:
        test_set.append(example)
      else:
        train_set.append(example)
    return train_set, test_set

original_size_test=len(test_set)
size_dev=int(round(original_size_test*0.5,0))
list_dev_indices=random.sample(range(original_size_test), size_dev)
new_dev_set=[]
new_test_set=[]
for i,example in enumerate(test_set):
  if i in list_dev_indices: new_dev_set.append(example)
  else: new_test_set.append(example)
new_train_set=train_set
random.shuffle(new_train_set)
random.shuffle(new_dev_set)
random.shuffle(new_test_set)

print ("TRAINING SET")
print ("Size training set: "+str(len(new_train_set)))
for example in new_train_set[:3]:
  print (example)
print ("    \n-------\n")
print ("DEV SET")
print ("Size development set: "+str(len(new_dev_set)))
for example in new_dev_set[:3]:
  print (example)
print ("    \n-------\n")
print ("TEST SET")
print ("Size test set: "+str(len(new_test_set)))
for example in new_test_set[:3]:
  print (example)


lemmatizer = nltk.stem.WordNetLemmatizer()
stopwords=set(nltk.corpus.stopwords.words('english'))
stopwords.add(".")
stopwords.add(",")
stopwords.add("--")
stopwords.add("``")


def get_list_tokens(string): # Function to retrieve the list of tokens from a string
  sentence_split=nltk.tokenize.sent_tokenize(string)
  list_tokens=[]
  for sentence in sentence_split:
    list_tokens_sentence=nltk.tokenize.word_tokenize(sentence)
    for token in list_tokens_sentence:
      list_tokens.append(lemmatizer.lemmatize(token).lower())
  return list_tokens

def get_vector_text(list_vocab,string):
  vector_text=np.zeros(len(list_vocab))
  list_tokens_string=get_list_tokens(string)
  for i, word in enumerate(list_vocab):
    if word in list_tokens_string:
      vector_text[i]=list_tokens_string.count(word)
  return vector_text

def get_vocabulary(training_set, num_features): # Function to retrieve vocabulary
  dict_word_frequency={}
  for instance in training_set:
    sentence_tokens=get_list_tokens(instance[0])
    for word in sentence_tokens:
      if word in stopwords: continue
      if word not in dict_word_frequency: dict_word_frequency[word]=1
      else: dict_word_frequency[word]+=1
  sorted_list = sorted(dict_word_frequency.items(), key=operator.itemgetter(1), reverse=True)[:num_features]
  vocabulary=[]
  for word,frequency in sorted_list:
    vocabulary.append(word)
  return vocabulary

def train_svm_classifier(training_set, vocabulary): # Function for training our svm classifier
  X_train=[]
  Y_train=[]
  for instance in training_set:
    vector_instance=get_vector_text(vocabulary,instance[0])
    X_train.append(vector_instance)
    Y_train.append(instance[1])
  # Finally, we train the SVM classifier
  svm_clf=sklearn.svm.SVC(kernel="linear",gamma='auto')
  svm_clf.fit(np.asarray(X_train),np.asarray(Y_train))
  return svm_clf


vocabulary=get_vocabulary(new_train_set, 1000)  # We use the get_vocabulary function to retrieve the vocabulary
svm_clf=train_svm_classifier(new_train_set, vocabulary) # We finally use the function to train our SVM classifier. This can take a while...

print (svm_clf.predict([get_vector_text(vocabulary,"unfortunately")]))

X_test=[]
Y_test=[]
for instance in new_test_set:
  vector_instance=get_vector_text(vocabulary,instance[0])
  X_test.append(vector_instance)
  Y_test.append(instance[1])
X_test=np.asarray(X_test)
Y_test_gold=np.asarray(Y_test)

from sklearn.metrics import classification_report
Y_text_predictions=svm_clf.predict(X_test)
print(classification_report(Y_test_gold, Y_text_predictions))

from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score

precision=precision_score(Y_test_gold, Y_text_predictions, average='macro')
recall=recall_score(Y_test_gold, Y_text_predictions, average='macro')
f1=f1_score(Y_test_gold, Y_text_predictions, average='macro')
accuracy=accuracy_score(Y_test_gold, Y_text_predictions)

print ("Precision: "+str(round(precision,3)))
print ("Recall: "+str(round(recall,3)))
print ("F1-Score: "+str(round(f1,3)))
print ("Accuracy: "+str(round(accuracy,3)))

from sklearn.metrics import confusion_matrix

print (confusion_matrix(Y_test_gold, Y_text_predictions))
