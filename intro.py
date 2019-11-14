import numpy as np
import nltk

nltk.download('punkt')
nltk.download('wordnet')

sentence1="Machine learning is the scientific study of algorithms and statistical models that computer systems use to perform a specific task without using explicit instructions, relying on patterns and inference instead. "
sentence2="It is seen as a subset of artificial intelligence. "
sentence3="Machine learning algorithms build a mathematical model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to perform the task. "
paragraph=sentence1+sentence2+sentence3
print ("Paragraph: " +str(paragraph))

list_tokens=nltk.tokenize.word_tokenize(paragraph)
print ("List_tokens: " +str(list_tokens))

sentence_split=nltk.tokenize.sent_tokenize(paragraph)
print ("Sentence_split: " +str(sentence_split))

list_sentence_tokens=[]
for sentence in sentence_split:
  list_sentence_tokens.append(nltk.tokenize.word_tokenize(sentence))
for sentence_tokens in list_sentence_tokens:
 print ("Sentence_tokens: "+str(sentence_tokens))
print ("List_sentence_tokens :" +str(list_sentence_tokens))


count_word = 0
for sentence_tokens in list_sentence_tokens:
    if "learning" in sentence_tokens:
      count_word += 1
print ("Number of sentences containing 'learning': " + str(count_word))


lemmatizer = nltk.stem.WordNetLemmatizer()
list_sentence_lemmas_lower=[]
for sentence_tokens in list_sentence_tokens:
  list_lemmas=[]
  for token in sentence_tokens:
    list_lemmas.append(lemmatizer.lemmatize(token).lower())
  list_sentence_lemmas_lower.append(list_lemmas)
print ("List_sentence_lemmas_lower: " +str(list_sentence_lemmas_lower))   #why the output with  u'algorithm',  u'prediction'u'decision
print ("List_lemmas :" +str(list_lemmas))


def get_list_tokens(string):
  sentence_split=nltk.tokenize.sent_tokenize(string)
  list_tokens=[]
  for sentence in sentence_split:
    list_tokens_sentence=nltk.tokenize.word_tokenize(sentence)
    for token in list_tokens_sentence:
      list_tokens.append(lemmatizer.lemmatize(token).lower())
  return list_tokens
print ("Get_list_tokens: " +str(get_list_tokens(paragraph)))

dict_freq_tokens={}
for sentence in list_sentence_lemmas_lower:
  for token in sentence:
    if token in dict_freq_tokens: dict_freq_tokens[token]+=1
    else: dict_freq_tokens[token]=1
vector_paragraph=np.zeros(len(dict_freq_tokens))
list_tokens=list(dict_freq_tokens.keys())
for i in range(len(list_tokens)):
  vector_paragraph[i]=dict_freq_tokens[list_tokens[i]]
 # print ("Dict_freq_tokens :" +str(dict_freq_tokens))
print ("Vector_paragraph: " +str(vector_paragraph))

dict_freq_tokens={}
count_sent=0
for sentence in list_sentence_lemmas_lower:
  count_sent+=1
  for token in sentence:
    if token in dict_freq_tokens: dict_freq_tokens[token]+=1
    else: dict_freq_tokens[token]=1
  for i in range(len(list_tokens)):
    token_vocab=list_tokens[i]
    if token_vocab in dict_freq_tokens: vector_paragraph[i]=dict_freq_tokens[token_vocab]
    else: vector_paragraph[i]=0
  print ("Sentence "+str(count_sent)+": "+str(vector_paragraph))
  dict_freq_tokens.clear()

#Exercise1
list_vocab = ['cat', 'dog', 'machine', 'field']
text = "Machine learning is a field where we study how machines learn."
def get_vector_text(list_vocab, string):   #define a new variable named get vec text(li,str)
    vector_text = np.zeros(len(list_vocab))
    list_tokens_string = get_list_tokens(string)
    for i, word in enumerate(list_vocab):
        if word in list_tokens_string:
            vector_text[i] = list_tokens_string.count(word)
    return vector_text
#print("Vector_text: " +str(get_vector_text))
#print("List_tokens_string: " +str(get_list_tokens))
print (get_vector_text(list_vocab, text))

#Exercise2
list_vocab=['cat','dog', 'machine', 'field']
#Cosine similarity
def cos_sim(a,b):
  return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
#Euclidean distance
def euc_dist(a,b):
  return np.linalg.norm(a-b)
string_a="Machine learning is a field where we study how machines learn."
string_b="The machine is not working."
vector_a=get_vector_text(list_vocab,string_a)
vector_b=get_vector_text(list_vocab,string_b)
print ("Cosine similarity: "+str(cos_sim(vector_a,vector_b)))
print ("Euclidean distance: "+str(euc_dist(vector_a,vector_b)))

def cos_sim(a,b):
  return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
def euc_dist(a,b):
  return np.linalg.norm(a-b)
string_a="My favorite animals are dogs and cats"
string_b="The machine is not working."
vector_a=get_vector_text(list_vocab,string_a)
vector_b=get_vector_text(list_vocab,string_b)
print ("Cosine similarity: "+str(cos_sim(vector_a,vector_b)))
print ("Euclidean distance: "+str(euc_dist(vector_a,vector_b)))


def cos_sim(a,b):
  return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
def euc_dist(a,b):
  return np.linalg.norm(a-b)
string_a="My favorite animals are dogs and cats"
string_b="What can we do with the cat and the dog? The cat is always fighting with the dog."
vector_a=get_vector_text(list_vocab,string_a)
vector_b=get_vector_text(list_vocab,string_b)
print ("Cosine similarity: "+str(cos_sim(vector_a,vector_b)))
print ("Euclidean distance: "+str(euc_dist(vector_a,vector_b)))

#a=np.array([1, 2, 3])
#b=np.array([10, 21, 32])

#print ("Cosine similarity: "+str(cos_sim(a,b)))
#print ("Euclidean distance: "+str(euc_dist(a,b)))

