#!/usr/bin/env python
# coding:utf-8

import numpy as np
import nltk
import sklearn
import operator
import requests
import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC

nltk.download('stopwords')

path_neg= '/Users/mayu/OneDrive - Cardiff University/applied machine learning/Coursework/datasets_coursework1/IMDb/train/imdb_train_neg.txt'
path_pos= '/Users/mayu/OneDrive - Cardiff University/applied machine learning/Coursework/datasets_coursework1/IMDb/train/imdb_train_pos.txt'

#import train dataset
df_pos= pd.read_csv(path_pos, delimiter= '\n', header=None)
df_neg= pd.read_csv(path_neg, delimiter= '\n', header=None)

# dataset_file_pos=open(path_pos).readlines()
# dataset_file_neg=open(path_neg).readlines()
#
#
# dataset_full=[]
# for pos_review in dataset_file_pos:
#   dataset_full.append((pos_review,1))
#   print (pos_review)
# for neg_review in dataset_file_neg:
#   dataset_full.append((neg_review,0))
#   print (neg_review)

df_pos.columns = ['text']
df_pos['label'] = '1'
df_neg.columns = ['text']
df_neg['label'] = '0'
dataset_full = pd.concat([df_pos,df_neg])

#print("df_pos: ", df_pos)
#print("df_neg: ", df_neg)

lemmatizer = nltk.stem.WordNetLemmatizer()
stopwords=set(nltk.corpus.stopwords.words('english'))
stopwords.add(".")
stopwords.add(",")
stopwords.add("--")
stopwords.add("``")
stopwords.add(":")
stopwords.add("/")
stopwords.add(">")
stopwords.add("<")
stopwords.add("them")
stopwords.add("me")
stopwords.add("he")
stopwords.add("she")
stopwords.add("about")
stopwords.add("ever")
stopwords.add("being")
stopwords.add("that")
stopwords.add("after")
stopwords.add("and")
stopwords.add("the")
stopwords.add("<>")
stopwords.add("are")
stopwords.add("how")
stopwords.add("do")
stopwords.add("to")
stopwords.add(" I")
stopwords.add("movie")
stopwords.add("really")
stopwords.add("e")
stopwords.add("n")
stopwords.add("h")
#
def get_list_tokens(string): # Function to retrieve the list of tokens from a string
  sentence_split=nltk.tokenize.sent_tokenize(string)
  list_tokens=[]
  for sentence in sentence_split:
    list_tokens_sentence=nltk.tokenize.word_tokenize(sentence)
    for token in list_tokens_sentence:
      list_tokens.append(lemmatizer.lemmatize(token).lower())
  return list_tokens

dict_word_frequency = {}
#for review in dataset_full['text']:
for pos_review in df_pos['text']:
    sentence_tokens = get_list_tokens(pos_review)
    for word in sentence_tokens:
        if word in stopwords: continue
        if word not in dict_word_frequency:
            dict_word_frequency[word] = 1
        else:
            dict_word_frequency[word] += 1

for neg_review in df_neg['text']:
    sentence_tokens = get_list_tokens(neg_review)
    for word in sentence_tokens:
        if word in stopwords: continue
        if word not in dict_word_frequency:
            dict_word_frequency[word] = 1
        else:
            dict_word_frequency[word] += 1

sorted_list = sorted(dict_word_frequency.items(), key=operator.itemgetter(1), reverse=True)[:1000]
i = 0
for word, frequency in sorted_list[:5]:
    i += 1
    print (str(i) + ". " + word + " - " + str(frequency))


# def lexical_diversity(text):
#   return len(set(text)) / len(text)
# lexical_diversity(pos_review)
# lexical_diversity(neg_review)



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
#
#
# filter words
def filter_words(data):
	sentence = []
	pos_set = set(['JJ', 'CC', 'RB', 'IN', 'NN'])
	# for sentence in list(data):
    # for word in sentence_tokens:
		sentence = word_tokenize(sentence)
		adj_words = []
		pos_tags = pos_tag(words)
		for word, pos in pos_tags:
			if pos in pos_set:
				adj_words.append(word)
		senstence.append(" ".join(adj_words))
	return words

tagged_sentence = nltk.corpus.treebank.tagged_sents(tagset='universal')
print("Number of Tagged Sentences ",len(tagged_sentence))
tagged_words=[tup for sent in tagged_sentence for tup in sent]
print("Total Number of Tagged words", len(tagged_words))
vocab=set([word for word,tag in tagged_words])
print("Vocabulary of the Corpus",len(vocab))
tags=set([tag for word,tag in tagged_words])
print("Number of Tags in the Corpus ",len(tags))


def features(sentence, index):
  ### sentence is of the form [w1,w2,w3,..], index is the position of the word in the sentence
  return {
    'is_first_capital': int(sentence[index][0].isupper()),
    'is_first_word': int(index == 0),
    'is_last_word': int(index == len(sentence) - 1),
    'is_complete_capital': int(sentence[index].upper() == sentence[index]),
    'prev_word': '' if index == 0 else sentence[index - 1],
    'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
    'is_numeric': int(sentence[index].isdigit()),
    'is_alphanumeric': int(bool((re.match('^(?=.*[0-9]$)(?=.*[a-zA-Z])', sentence[index])))),
    'prefix_1': sentence[index][0],
    'prefix_2': sentence[index][:2],
    'prefix_3': sentence[index][:3],
    'prefix_4': sentence[index][:4],
    'suffix_1': sentence[index][-1],
    'suffix_2': sentence[index][-2:],
    'suffix_3': sentence[index][-3:],
    'suffix_4': sentence[index][-4:],
    'word_has_hyphen': 1 if '-' in sentence[index] else 0
  }

def untag(sentence):
  return [word for word, tag in sentence]


def prepareData(tagged_sentences):
  X, y = [], []
  for sentences in tagged_sentences:
    X.append([features(untag(sentences), index) for index in range(len(sentences))])
    y.append([tag for word, tag in sentences])
  return X, y

X_train, y_train = prepareData(train_set)
X_test, y_test = prepareData(test_set)


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

vocabulary=get_vocabulary(new_train_set, 100)  # We use the get_vocabulary function to retrieve the vocabulary
svm_clf=train_svm_classifier(new_train_set, vocabulary) # We finally use the function to train our SVM classifier. This can take a while...

print (svm_clf.predict([get_vector_text(vocabulary,"unfortunately")]))
#
path_neg_test= '/Users/mayu/OneDrive - Cardiff University/applied machine learning/Coursework/datasets_coursework1/IMDb/test/imdb_test_neg.txt'
path_pos_test= '/Users/mayu/OneDrive - Cardiff University/applied machine learning/Coursework/datasets_coursework1/IMDb/test/imdb_test_pos.txt'

#dataset_file_pos= open(path_pos).readline()
df_pos_test= pd.read_csv(path_pos_test, delimiter= '\n', header=None)

#dataset_file_neg= open(path_neg).readline()
df_neg_test= pd.read_csv(path_neg_test, delimiter= '\n', header=None)

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


path_neg_dev= '/Users/mayu/OneDrive - Cardiff University/applied machine learning/Coursework/datasets_coursework1/IMDb/dev/imdb_dev_neg.txt'
path_pos= '/Users/mayu/OneDrive - Cardiff University/applied machine learning/Coursework/datasets_coursework1/IMDb/dev/imdb_dev_pos.txt'

#dataset_file_pos= open(path_pos).readline()
df_pos_dev= pd.read_csv(path_pos_dev, delimiter= '\n', header=None)

#dataset_file_neg= open(path_neg).readline()
df_neg= pd.read_csv(path_neg_dev, delimiter= '\n', header=None)


Y_dev=[]
for instance in new_dev_set:
  Y_dev.append(instance[1])
Y_dev_gold=np.asarray(Y_dev)

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


