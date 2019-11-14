#!/usr/bin/env python
# coding:utf-8

import numpy as np
import nltk
import sklearn
import operator
import requests
from nltk.corpus import stopwords

nltk.download('stopwords') # If needed
nltk.download('punkt') # If needed
nltk.download('wordnet') # If needed

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

#Method 1
response = requests.get(url)
dataset_file = response.text.split("\n")

#Method 2 - Google Colab
##from google.colab import drive
##drive.mount('/content/drive/')
##path= '/content/drive/My Drive/pima-indians-diabetes.data.csv'
##dataset_file=open(path).readlines()


#Method 3 - Local
##path='/home/user/Downloads/pima-indians-diabetes.data.csv'
##dataset_file=open(path).readlines()

#print ("Number of patients: "+str(len(dataset_file))+"\n")
count_line=0
for patient_line in dataset_file:
    count_line +=1
for patient_line in dataset_file[:5]:
  print ("Patient_line: " +str(patient_line))

X_train=[]
Y_train=[]
for patient_line in dataset_file:
  patient_linesplit=patient_line.split(",")
  vector_patient_features=np.zeros(len(patient_linesplit)-1)
  for i in range(len(patient_linesplit)-1):
    vector_patient_features[i]=float(patient_linesplit[i])
  X_train.append(vector_patient_features)
  Y_train.append(int(patient_linesplit[-1]))

X_train_diabetes=np.asarray(X_train)
Y_train_diabetes=np.asarray(Y_train) # This step is really not necessary, but it is recommended to work with numpy arrays instead of Python lists.

svm_clf_diabetes=sklearn.svm.SVC(gamma='auto') # Initialize the SVM model
svm_clf_diabetes.fit(X_train_diabetes,Y_train_diabetes) # Train the SVM model

patient_1=['0', '100', '86', '20', '39', '35.1', '0.242', '21']
patient_2=['1', '10', '70', '45', '543', '30.5', '0.158', '51']
print (svm_clf_diabetes.predict([patient_1]))
print (svm_clf_diabetes.predict([patient_2]))

X_train=[]
Y_train=[]
#To complete
selected_features = [0, 5, 7]  # Number of times pregnant, Body mass index and Age
for patient_line in dataset_file:
    patient_linesplit = patient_line.split(",")
    vector_patient_features = np.zeros(len(selected_features))
    feature_index = 0
    for i in range(len(patient_linesplit) - 1):
        if i in selected_features:
            vector_patient_features[feature_index] = float(patient_linesplit[i])
            feature_index += 1
    X_train.append(vector_patient_features)
    Y_train.append(int(patient_linesplit[-1]))

svm_clf = sklearn.svm.SVC(kernel="linear", gamma='auto')  # Load the (linear) SVM model
svm_clf.fit(X_train, Y_train)  # Train the SVM model

patient_1 = ['3', '35.2', '500']
patient_2 = ['1', '20.5', '0']
print (svm_clf.predict([patient_1]))
print (svm_clf.predict([patient_2]))



url_pos="http://josecamachocollados.com/rt-polarity.pos.txt" # Containing all positive reviews, one review per line
url_neg="http://josecamachocollados.com/rt-polarity.neg.txt" # Containing all negative reviews, one review per line
#Load positive reviews
response_pos = requests.get(url_pos)
dataset_file_pos = response_pos.text.split("\n")

#Load negative reviews
response_neg = requests.get(url_neg)
dataset_file_neg = response_neg.text.split("\n")

print ("Positive reviews:\n")
for pos_review in dataset_file_pos[:5]:
    print (pos_review)
print ("\n   ------\n")
print ("Negative reviews:\n")
for neg_review in dataset_file_neg[:5]:
    print (neg_review)

lemmatizer = nltk.stem.WordNetLemmatizer()

# Function taken from Session 1
def get_list_tokens(string):
  sentence_split=nltk.tokenize.sent_tokenize(string)
  list_tokens=[]
  for sentence in sentence_split:
    list_tokens_sentence=nltk.tokenize.word_tokenize(sentence)
    for token in list_tokens_sentence:
      list_tokens.append(lemmatizer.lemmatize(token).lower())
  return list_tokens


# First, we get the stopwords list from nltk
stopwords = set(nltk.corpus.stopwords.words('english'))
#for word in stopwords:
#   print (word)
#input()

# We can add more words to the stopword list, like punctuation marks
stopwords.add(".")
stopwords.add(",")
stopwords.add("--")
stopwords.add("``")

#print(len(stopWords))
#print(stopWords)

# Now we create a frequency dictionary with all words in the dataset
# This can take a few minutes depending on your computer, since we are processing more than ten thousand sentences

dict_word_frequency = {}
for pos_review in dataset_file_pos:
    sentence_tokens = get_list_tokens(pos_review)
    for word in sentence_tokens:
        if word in stopwords: continue
        if word not in dict_word_frequency:
            dict_word_frequency[word] = 1
        else:
            dict_word_frequency[word] += 1
for neg_review in dataset_file_neg:
    sentence_tokens = get_list_tokens(neg_review)
    for word in sentence_tokens:
        if word in stopwords: continue
        if word not in dict_word_frequency:
            dict_word_frequency[word] = 1
        else:
            dict_word_frequency[word] += 1

# Now we create a sorted frequency list with the top 1000 words, using the function "sorted". Let's see the 15 most frequent words
sorted_list = sorted(dict_word_frequency.items(), key=operator.itemgetter(1), reverse=True)[:1000]
i = 0
for word, frequency in sorted_list[:15]:
    i += 1
    print (str(i) + ". " + word + " - " + str(frequency))

# Finally, we create our vocabulary based on the sorted frequency list
vocabulary = []
for word, frequency in sorted_list:
    vocabulary.append(word)

def get_vector_text(list_vocab,string):
  vector_text=np.zeros(len(list_vocab))
  list_tokens_string=get_list_tokens(string)
  for i, word in enumerate(list_vocab):
    if word in list_tokens_string:
      vector_text[i]=list_tokens_string.count(word)
  return vector_text

# This can take a while, as we are converting more than ten thousand sentences into vectors!
X_train=[]
Y_train=[]
for pos_review in dataset_file_pos:
  vector_pos_review=get_vector_text(vocabulary,pos_review)
  X_train.append(vector_pos_review)
  Y_train.append(1)
for neg_review in dataset_file_neg:
  vector_neg_review=get_vector_text(vocabulary,neg_review)
  X_train.append(vector_neg_review)
  Y_train.append(0)

X_train_sentanalysis=np.asarray(X_train)
Y_train_sentanalysis=np.asarray(Y_train)

svm_clf_sentanalysis=sklearn.svm.SVC(kernel='linear',gamma='auto')
svm_clf_sentanalysis.fit(X_train_sentanalysis,Y_train_sentanalysis) # Train the SVM model. This may also take a while.

sentence_1="It was fascinating, probably one of the best movies I've ever seen."
sentence_2="unfortunately the story and the actors are served with a hack script ."
sentence_3="Bad movie, probably one of the worst I have ever seen."

print (svm_clf_sentanalysis.predict([get_vector_text(vocabulary,sentence_1)]))
print (svm_clf_sentanalysis.predict([get_vector_text(vocabulary,sentence_2)]))
print (svm_clf_sentanalysis.predict([get_vector_text(vocabulary,sentence_3)]))

def get_vocabulary(dataset_file_pos, dataset_file_neg, num_features): # Function to retrieve vocabulary
  dict_word_frequency={}
  for pos_review in dataset_file_pos:
    sentence_tokens=get_list_tokens(pos_review)
    for word in sentence_tokens:
      if word in stopwords: continue
      if word not in dict_word_frequency: dict_word_frequency[word]=1
      else: dict_word_frequency[word]+=1
  for neg_review in dataset_file_neg:
    sentence_tokens=get_list_tokens(neg_review)
    for word in sentence_tokens:
      if word in stopwords: continue
      if word not in dict_word_frequency: dict_word_frequency[word]=1
      else: dict_word_frequency[word]+=1
  sorted_list = sorted(dict_word_frequency.items(), key=operator.itemgetter(1), reverse=True)[:num_features]
  vocabulary=[]
  for word,frequency in sorted_list:
    vocabulary.append(word)
  return vocabulary

#def train_svm_classifier(dataset_file_pos, dataset_file_neg, x):
  #To complete
  def train_svm_classifier(dataset_file_pos, dataset_file_neg, vocabulary):
      # First we convert sentences to vectors, which will be used as features
      X_train = []
      Y_train = []
      for pos_review in dataset_file_pos:
          vector_pos_review = get_vector_text(vocabulary, pos_review)
          X_train.append(vector_pos_review)
          Y_train.append(1)
      for neg_review in dataset_file_neg:
          vector_neg_review = get_vector_text(vocabulary, neg_review)
          X_train.append(vector_neg_review)
          Y_train.append(0)
      X_train_sentanalysis = np.asarray(X_train)
      Y_train_sentanalysis = np.asarray(Y_train)
      # Finally, we train the SVM classifier
      svm_clf = sklearn.svm.SVC(kernel="linear", gamma='auto')
      svm_clf.fit(X_train_sentanalysis, Y_train_sentanalysis)
      return svm_clf

  new_vocabulary = get_vocabulary(dataset_file_pos, dataset_file_neg,
                                  1200)  # We use the get_vocabulary function to retrieve the vocabulary
  svm_clf = train_svm_classifier(dataset_file_pos, dataset_file_neg, new_vocabulary)
  print (svm_clf.predict([get_vector_text(new_vocabulary, sentence_1)]))
  print (svm_clf.predict([get_vector_text(new_vocabulary, sentence_2)]))
  print (svm_clf.predict([get_vector_text(new_vocabulary, sentence_3)]))



from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

fs_sentanalysis=SelectKBest(chi2, k=500).fit(X_train_sentanalysis, Y_train_sentanalysis)
X_train_sentanalysis_new = fs_sentanalysis.transform(X_train_sentanalysis)
#X_train_new = SelectKBest(chi2, k=500).fit_transform(X_train, Y_train)
print ("Size original training matrix: "+str(X_train_sentanalysis.shape))
print ("Size new training matrix: "+str(X_train_sentanalysis_new.shape))

svm_clf_sentanalysis_=sklearn.svm.SVC(gamma='auto') # Change the name here, e.g. 'new sentanalysis_svm_clf', and below if you don't want to replace your old classifier.
svm_clf_sentanalysis_.fit(X_train_sentanalysis_new,Y_train_sentanalysis) #Train the new SVM model. This may take a while.
sentence_3="Highly recommended: I enjoyed the movie from the beginning to the end."
sentence_4="I got a bit bored, it was not what I was expecting."
print (svm_clf_sentanalysis_.predict(fs_sentanalysis.transform([get_vector_text(vocabulary,sentence_3)])))
print (svm_clf_sentanalysis_.predict(fs_sentanalysis.transform([get_vector_text(vocabulary,sentence_4)])))

fs_diabetes=SelectKBest(chi2, k=7).fit(X_train_diabetes, Y_train_diabetes)
X_train_diabetes_new = fs_diabetes.transform(X_train_diabetes)

svm_clf_diabetes=sklearn.svm.SVC(kernel="linear",gamma='auto')
svm_clf_diabetes.fit(X_train_diabetes_new,Y_train_diabetes)
patient_1=['0', '100', '86', '20', '39', '35.1', '0.242', '21']
patient_2=['1', '197', '70', '45', '543', '30.5', '0.158', '51']
print (svm_clf_diabetes.predict(fs_diabetes.transform([patient_1])))
print (svm_clf_diabetes.predict(fs_diabetes.transform([patient_2])))

