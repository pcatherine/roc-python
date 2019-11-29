import pickle
from nltk import word_tokenize
from nltk.classify import ClassifierI
from statistics import mode
import json

# Ensemble model
class EnsembleClassifier(ClassifierI):
    
    def __init__(self, *classifiers):
        self._classifiers = classifiers
    
    # Retorna qual classificador será usado, baseado na melhor acurácia
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    # Retorna a confiança na classificação 
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf 

# Carrega os modelos pré-treinados
def load_model(file_path): 
    classifier_file = open(file_path, "rb")
    classifier = pickle.load(classifier_file)
    classifier_file.close()
    return classifier

# Load de palavras e features do arquivo de treinamento
pck = open('word_features.pickle','rb')
word_features = pickle.load(pck)

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for word in word_features:
        features[word] = (word in words)
    return features

# Função para análise de sentimento e confiança
def sentiment_analysis(comments):
    feats = []
    classification = []
    confidence = []
    
    # Extrai as features para cada review
    for line in comments:
        feats.append(find_features(line['review']))
    
    # Volta a classificação e confiança para cada review
    for feat in feats:
        classification.append(ensemble.classify(feat))
        confidence.append(ensemble.confidence(feat))
        # class_dict.update({"classification":ensemble.classify(feat), "confidence":ensemble.confidence(feat)})

    lenght_class = len(classification)
    cont_p, cont_n = 0, 0
    for cl in classification:
        if cl == 'pos': 
            cont_p += 1
        if cl == 'neg': 
            cont_n += 1
    res1 = cont_p/lenght_class
    res2 = cont_n/lenght_class

    return res1, res2
    # Retorno de cada classificador
    # return log_reg.classify(feats
    # return onb.classify(feats)
    # return rf.classify(feats)
    # return sgd.classify(feats)
    # return svc.classify(feats)

# Logistic Regression 
log_reg = load_model('saved_files/log_reg.pickle')

# Original Naive Bayes
onb = load_model('saved_files/onb.pickle')

# Random Forest 
rf = load_model('saved_files/rf.pickle')

# Stochastic Gradient Descent
sgd = load_model('saved_files/sgd.pickle')

# Support Vector Classifier 
svc = load_model('saved_files/svc.pickle')

# Ensemble classifier 
ensemble = EnsembleClassifier(log_reg, onb, rf, sgd, svc)

if __name__ == "__main__" : 
    comments = [
        {"review":"This new show popped up on Netflix so we decided to watch an episode, 5 hours later the whole season came to an end. Really enjoyable dark comedy, drama, crime all wrapped up in to one. Great performances from both the ladies. If you are looking for the 'who done it?' type of show you found it, however there is a slight twist. How does one reveal what they know. Enjoy it, looking forward to season 2."},
        {"review":"I know too much about grief and this story explains so many different stages of grief that are not talked about. The story has so many twist and turns and it's well worth the watch"},
        {"review":"The worst show i have ever seen"},
        {"review":"You can really see great acting come out with a good script, dialogue, and story line. Christina Applegate is so good this! You literally never know what's going to happen next in this show. It's a binger!"},
        {"review":"It was bad"}
    ]
    print(sentiment_analysis(comments))