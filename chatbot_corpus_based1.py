#########################################################################################################################################
                                   ## Generate Bot responses using Cosine similarity_scores ##
#########################################################################################################################################

# importing necessary library dependencies
import nltk
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import pickle

# Ignoring the warnings
import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

import spacy
nlp = spacy.load('en_core_web_sm')

import string
punct = string.punctuation
from spacy.lang.en.stop_words import STOP_WORDS
stopwords = list(STOP_WORDS) # list of stopwords

# creating a function for data cleaning
def text_data_cleaning(sentence):
  doc = nlp(sentence)

  tokens = [] # list of tokens
  for token in doc:
    if token.lemma_ != "-PRON-":
      temp = token.lemma_.lower().strip()
    else:
      temp = token.lower_
    tokens.append(temp)
 
  cleaned_tokens = []
  for token in tokens:
    if token not in stopwords and token not in punct:    # Stopwords and punctuation removal
      cleaned_tokens.append(token)
  return cleaned_tokens

import pandas as pd
import numpy as np
ent = np.array(pd.read_csv("E://#Proactive Interaction Manager/robot's responses/ent.csv"))
sugg = np.array(pd.read_csv("E://#Proactive Interaction Manager/robot's responses/sugg.csv"))
remind = np.array(pd.read_csv("E://#Proactive Interaction Manager/robot's responses/remind.csv"))
others = np.array(pd.read_csv("E://#Proactive Interaction Manager/robot's responses/others.csv"))


# loading the saved model
# load the saved model from file using pickle
with open("E://#Proactive Interaction Manager/robot's responses/rf_model.pkl", 'rb') as f:
    rf = pickle.load(f)

# getting input from the user
user_input = input("\nUser : ")

def make_list(array):
    new = []
    for i in array:
        new.append(i[0])
    return new

def directing_classes(classes):
   if classes[0] == "entertainment":
      return (make_list(ent.tolist()))
   elif classes[0] == "suggestions":
      return (make_list(sugg.tolist()))
   elif classes[0] == "reminders":
      return (make_list(remind.tolist()))
   else:
      return (make_list(others.tolist()))
   

# apologizing
def appol(sim_score, selected_response):
   if max(sim_score) < 0.09:
      print ("Sorry, I couldn't understand what you are asking")
      #print (f"Max similarity score is : {max(similarity_scores)}\n")
   else:
      # printing the selected response
        print (f"MIRob : {selected_response}\n")
        #print (f"Max similarity score is : {max(similarity_scores)}\n")
      

# looping through getting  user inputs
while user_input:
    # checking if the user want to quit otherwise continue
    if user_input.lower() == "quit":
        print (f"MIRob : Thank you !, Feel free to ask me whenever necessary !!!\n")
        break
    else:
        user_input = [user_input] 
        classes = rf.predict(user_input)
        # Preprocess the response sentences and user input sentence
        response_sentences = directing_classes(classes)
        #print (response_sentences)
        user_input = user_input[0]
        #user_input = input("Enter the user response : ")
        lemmatizer = nltk.stem.WordNetLemmatizer()
        remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
        response_tokens = [nltk.word_tokenize(sent.lower().translate(remove_punct_dict)) for sent in response_sentences]
        user_input_tokens = nltk.word_tokenize(user_input.lower().translate(remove_punct_dict))
        response_lemmatized = [[lemmatizer.lemmatize(word) for word in tokens] for tokens in response_tokens]
        user_input_lemmatized = [lemmatizer.lemmatize(word) for word in user_input_tokens]

        # Convert the preprocessed sentences to numerical vectors using TF-IDF
        corpus = [' '.join(sent) for sent in response_lemmatized]
        vectorizer = TfidfVectorizer()
        response_vectors = vectorizer.fit_transform(corpus)
        user_input_vector = vectorizer.transform([' '.join(user_input_lemmatized)])

        # Calculate cosine similarity between user input and response sentences
        similarity_scores = cosine_similarity(user_input_vector, response_vectors)[0]

        # Select the response with the highest cosine similarity score
        index_of_max_score = similarity_scores.argmax()
        selected_response = response_sentences[index_of_max_score]

        # printing response
        appol(similarity_scores, selected_response)

        # getting again user input
        user_input = input("User : ")

    


