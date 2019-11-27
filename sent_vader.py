### Projeto Final - Computação Afetiva ###
### VADER Sentiment Analyser ###

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import flask

app = flask.Flask(__name__)
app.config["DEBUG"] = True

# Rota que retorna a análise de sentimento
@app.route('/sentiment/vader', methods=['POST'])
def sentiment_scores(sentence): 
  
    # Objeto SentimentIntensityAnalyzer 
    analyser = SentimentIntensityAnalyzer()
 
    sent_dict = analyser.polarity_scores(sentence) 
  
    # Retorna análise geral da frase
    if sent_dict['compound'] >= 0.05 : 
        # Positivo
        predict = 1
  
    elif sent_dict['compound'] <= - 0.05 : 
        # Negativo
        predict = -1
        
    else : 
        # Neutro
        predict = 0

    return predict

app.run()


if __name__ == "__main__" : 

    # Teste
    sentence = "Chernobyl is the BEST tv show i have ever seen!!!!!!! :D It is amaaaaaazing!!!!!"

    if sentiment_scores(sentence) == 1:
        print('Positivo')
    elif sentiment_scores(sentence) == -1:
        print('Negativo')
    else:
        print('Neutro')
