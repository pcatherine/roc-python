import nltk
import random
import re
import os
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

# Leitura de dataset rotulado
# Modifique o path para o seu diretório
files_pos = os.listdir('dataset/train/pos')
files_pos = [open('dataset/train/pos/'+f, 'r', encoding="utf8").read() for f in files_pos]
files_neg = os.listdir('dataset/train/neg')
files_neg = [open('dataset/train/neg/'+f, 'r', encoding="utf8").read() for f in files_neg]

# Baixando corpora necessários
# Descomente para rodar a primeira vez
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')
# nltk.download('punkt')

all_words = []
documents = []

# Criando lista de stopwords
stop_words = list(set(stopwords.words('english')))

# J - adjetivo
# R - advérbio
# V - verbo
bag_of_words = ["J"]

for p in files_pos:
    # Arquivos com rótulo positivo
    # Criando uma tupla com dois elementos, o primeiro é a review e o segundo o rótulo
    documents.append((p, "pos"))
    
    # Removendo pontuação
    cleaned = re.sub(r'[^(a-zA-Z)\s]','', p)
    
    # Tokenização 
    tokenized = word_tokenize(cleaned)
    
    # Removendo stopwords 
    stopped = [word for word in tokenized if not word in stop_words]
    
    # PoS-tagging
    pos = nltk.pos_tag(stopped)
    
    # Criando o bag of words dos adjetivos permitidos
    for word in pos:
        if word[1][0] in bag_of_words:
            all_words.append(word[0].lower())
   
for n in files_neg:
    # Arquivos com rótulo negativo
    # Criando uma tupla com dois elementos, o primeiro é a review e o segundo o rótulo
    documents.append((n, "neg"))

    # Removendo pontuação
    cleaned = re.sub(r'[^(a-zA-Z)\s]','', n)

    # Tokenização
    tokenized = word_tokenize(cleaned)

    # Removendo stopwords
    stopped = [word for word in tokenized if not word in stop_words]

    # PoS-tagging
    neg = nltk.pos_tag(stopped)

    # Criando o bag of words dos adjetivos permitidos
    for word in neg:
        if word[1][0] in bag_of_words:
            all_words.append(word[0].lower())

# print("PASSOU")

# TF-IDF
all_words = nltk.FreqDist(all_words)

# Lista das 5000 palavras mais frequentes
word_features = list(all_words.keys())[:5000]
pck = open("word_features.pickle", "wb")
pickle.dump(word_features,pck)
pck.close()

# print("PASSOU2")

# Criando um dicionário de features
def find_features(document):
    words = word_tokenize(document)
    features = {}
    for word in word_features:
        features[word] = (word in words)
    return features

# Features para cada review
feature_set = [(find_features(rev), category) for (rev, category) in documents]
 
random.shuffle(feature_set)

training_set = feature_set[:20000]
testing_set = feature_set[20000:]

# print(TERMINOU)

# Descomente para treinar modelo
# log_reg = SklearnClassifier(LogisticRegression())
# log_reg.train(training_set)
# p1 = open("log_reg.pickle", "wb")
# pickle.dump(log_reg,p1)
# p1.close()
# print("Classifier accuracy percent:",(nltk.classify.accuracy(log_reg, testing_set))*100)

# Descomente para treinar modelo
# onb = nltk.NaiveBayesClassifier.train(training_set)
# p2 = open("onb.pickle", "wb")
# pickle.dump(onb,p2)
# p2.close()

# Printando acurácia e features
# print("Classifier accuracy percent:",(nltk.classify.accuracy(onb, testing_set))*100)
# classifier.show_most_informative_features(10)

# Descomente para treinar modelo
# rf = SklearnClassifier(RandomForestClassifier())
# rf.train(training_set)
# p3 = open("rf.pickle", "wb")
# pickle.dump(rf,p3)
# p3.close()
# print("Classifier accuracy percent:",(nltk.classify.accuracy(rf, testing_set))*100)

# Descomente para treinar modelo
# sgd = SklearnClassifier(SGDClassifier())
# sgd.train(training_set)
# p4 = open("sgd.pickle", "wb")
# pickle.dump(sgd,p4)
# p4.close()
# print("Classifier accuracy percent:",(nltk.classify.accuracy(sgd, testing_set))*100)

# Descomente para treinar modelo
# svc = SklearnClassifier(SVC())
# svc.train(training_set)
# p5 = open("svc.pickle", "wb")
# pickle.dump(svc,p5)
# p5.close()
# print("Classifier accuracy percent:",(nltk.classify.accuracy(svc, testing_set))*100)