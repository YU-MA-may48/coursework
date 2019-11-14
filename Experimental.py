#!/usr/bin/env python
# coding:utf-8

import numpy as np
import nltk
import sklearn
import operator
import requests
nltk.download('stopwords') # If needed
nltk.download('punkt') # If needed
nltk.download('wordnet') # If needed

url_pos="http://josecamachocollados.com/rt-polarity.pos.txt" # Containing all positive reviews, one review per line
url_neg="http://josecamachocollados.com/rt-polarity.neg.txt" # Containing all negative reviews, one review per line

#Load positive reviews
response_pos = requests.get(url_pos)
dataset_file_pos = response_pos.text.split("\n")

#Load negative reviews
response_neg = requests.get(url_neg)
dataset_file_neg = response_neg.text.split("\n")

dataset_full=[]
for pos_review in dataset_file_pos:
  dataset_full.append((pos_review,1))
for neg_review in dataset_file_neg:
  dataset_full.append((neg_review,0))


from sklearn.model_selection import train_test_split
import random

size_dataset_full=len(dataset_full)
size_test=int(round(size_dataset_full*0.2,0))
#print (size_dataset_full)
#print (size_test)

list_test_indices=random.sample(range(size_dataset_full), size_test)
train_set=[]
test_set=[]
for i,example in enumerate(dataset_full):
  if i in list_test_indices: test_set.append(example)
  else: train_set.append(example)


random.shuffle(train_set)
random.shuffle(test_set)

#def get_train_test_split(dataset_full,ratio):
  #pre_train_set=[]
  #pre_test_set=[]
  # To complete...
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


  # To verify
train_set_, test_set_ = get_train_test_split(dataset_full, 0.2)
#print ("Size training set: " + str(len(train_set_)))
#print ("Size test set: " + str(len(test_set_)))


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

# Function taken from Session 1
def get_list_tokens(string): # Function to retrieve the list of tokens from a string
  sentence_split=nltk.tokenize.sent_tokenize(string)
  list_tokens=[]
  for sentence in sentence_split:
    list_tokens_sentence=nltk.tokenize.word_tokenize(sentence)
    for token in list_tokens_sentence:
      list_tokens.append(lemmatizer.lemmatize(token).lower())
  return list_tokens

#print (sentence_split)
#print (list_tokens_sentence)

# Function taken from Session 2
def get_vector_text(list_vocab,string):
  vector_text=np.zeros(len(list_vocab))
  list_tokens_string=get_list_tokens(string)
  for i, word in enumerate(list_vocab):
    if word in list_tokens_string:
      vector_text[i]=list_tokens_string.count(word)
  return vector_text
#print (list_vocab)
#print (vector_text)


# Functions slightly modified from Session 2
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
#print (training_set)
#print (num_features)


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

# We first get the gold standard labels from the development set

Y_dev=[]
for instance in new_dev_set:
  Y_dev.append(instance[1])
Y_dev_gold=np.asarray(Y_dev)

# Now we can train our three models with the different number of features, and test each of them in the dev set

list_num_features=[250,500,750,1000]
best_accuracy_dev=0.0
for num_features in list_num_features:
  # First, we get the vocabulary from the training set and train our svm classifier
  vocabulary=get_vocabulary(new_train_set, num_features)
  svm_clf=train_svm_classifier(new_train_set, vocabulary)
  # Then, we transform our dev set into vectors and make the prediction on this set
  X_dev=[]
  for instance in new_dev_set:
    vector_instance=get_vector_text(vocabulary,instance[0])
    X_dev.append(vector_instance)
  X_dev=np.asarray(X_dev)
  Y_dev_predictions=svm_clf.predict(X_dev)
  # Finally, we get the accuracy results of the classifier
  accuracy_dev=accuracy_score(Y_dev_gold, Y_dev_predictions)
  print ("Accuracy with "+str(num_features)+": "+str(round(accuracy_dev,3)))
  if accuracy_dev>=best_accuracy_dev:
    best_accuracy_dev=accuracy_dev
    best_num_features=num_features
    best_vocabulary=vocabulary
    best_svm_clf=svm_clf
print ("\n Best accuracy overall in the dev set is "+str(round(best_accuracy_dev,3))+" with "+str(best_num_features)+" features.")

X_test=[]
Y_test=[]
for instance in new_test_set:
  vector_instance=get_vector_text(best_vocabulary,instance[0])
  X_test.append(vector_instance)
  Y_test.append(instance[1])
best_X_test=np.asarray(X_test)
Y_test_gold=np.asarray(Y_test)

best_Y_text_predictions=best_svm_clf.predict(best_X_test)
print(classification_report(Y_test_gold, best_Y_text_predictions))

#list_num_features=[100,500,1000]
# To complete
list_num_features=[100,500,1000]
best_f1_dev=0.0
for num_features in list_num_features:
  vocabulary=get_vocabulary(new_train_set, num_features)
  svm_clf=train_svm_classifier(new_train_set, vocabulary)
  X_dev=[]
  for instance in new_dev_set:
    vector_instance=get_vector_text(vocabulary,instance[0])
    X_dev.append(vector_instance)
  X_dev=np.asarray(X_dev)
  Y_dev_predictions=svm_clf.predict(X_dev)
  f1_dev=f1_score(Y_dev_gold, Y_dev_predictions, average='macro')
  print ("F1-Score with "+str(num_features)+": "+str(round(f1_dev,3)))
  if f1_dev>=best_f1_dev:
    best_f1_dev=f1_dev
    best_num_features=num_features
    best_vocabulary=vocabulary
    best_svm_clf=svm_clf
print ("\nBest F-Score overall in the dev set is "+str(round(best_f1_dev,3))+" with "+str(best_num_features)+" features.")
# Now we test the best classifier (in the dev set) on the test set
X_test=[]
Y_test=[]
for instance in new_test_set:
  vector_instance=get_vector_text(best_vocabulary,instance[0])
  X_test.append(vector_instance)
  Y_test.append(instance[1])
Y_test_gold=np.asarray(Y_test)
best_X_test=np.asarray(X_test)
best_Y_text_predictions=best_svm_clf.predict(best_X_test)
print("\nPerformance in the test set\n")
print(classification_report(Y_test_gold, best_Y_text_predictions))


from sklearn.model_selection import KFold

kf = KFold(n_splits=5)
random.shuffle(dataset_full)
kf.get_n_splits(dataset_full)
for train_index, test_index in kf.split(dataset_full):
  train_set_fold=[]
  test_set_fold=[]
  accuracy_total=0.0
  for i,instance in enumerate(dataset_full):
    if i in train_index:
      train_set_fold.append(instance)
    else:
      test_set_fold.append(instance)
  vocabulary_fold=get_vocabulary(train_set_fold, 500)
  svm_clf_fold=train_svm_classifier(train_set_fold, vocabulary_fold)
  X_test_fold=[]
  Y_test_fold=[]
  for instance in test_set_fold:
    vector_instance=get_vector_text(vocabulary_fold,instance[0])
    X_test_fold.append(vector_instance)
    Y_test_fold.append(instance[1])
  Y_test_fold_gold=np.asarray(Y_test_fold)
  X_test_fold=np.asarray(X_test_fold)
  Y_test_predictions_fold=svm_clf_fold.predict(X_test_fold)
  accuracy_fold=accuracy_score(Y_test_fold_gold, Y_test_predictions_fold)
  accuracy_total+=accuracy_fold
  print ("Fold completed.")
average_accuracy=accuracy_total/5
print ("\nAverage Accuracy: "+str(round(accuracy_fold,3)))

num_folds = 3
num_features = 1000
kf = KFold(n_splits=num_folds)
random.shuffle(dataset_full)
kf.get_n_splits(dataset_full)
j_fold = 0
accuracy_total = 0.0
for train_index, test_index in kf.split(dataset_full):
  j_fold += 1
  train_set_fold = []
  test_set_fold = []
  for i, instance in enumerate(dataset_full):
    if i in train_index:
      train_set_fold.append(instance)
    else:
      test_set_fold.append(instance)
  vocabulary_fold = get_vocabulary(train_set_fold, num_features)
  svm_clf_fold = train_svm_classifier(train_set_fold, vocabulary_fold)
  X_test_fold = []
  Y_test_fold = []
  for instance in test_set_fold:
    vector_instance = get_vector_text(vocabulary_fold, instance[0])
    X_test_fold.append(vector_instance)
    Y_test_fold.append(instance[1])
  Y_test_fold_gold = np.asarray(Y_test_fold)
  X_test_fold = np.asarray(X_test_fold)
  Y_test_predictions_fold = svm_clf_fold.predict(X_test_fold)
  accuracy_fold = accuracy_score(Y_test_fold_gold, Y_test_predictions_fold)
  accuracy_total += accuracy_fold
  print ("Fold " + str(j_fold) + "/" + str(num_folds) + " completed. Accuracy: " + str(accuracy_fold))

average_accuracy = accuracy_total / num_folds
print ("\nAverage Accuracy: " + str(round(average_accuracy, 3)))


