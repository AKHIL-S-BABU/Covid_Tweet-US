import streamlit as st
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import joblib
#nltk.download('stopwords')
st.title('Covid Tweet-US Classifier')
#punctuation = ["""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""]
    
#@st.cache
#def remove_punct_stopwords(message):
 #   form_str = [char for char in message if char not in punctuation]
  #  form_str_join = ''.join(form_str)
   # words_stop = nltk.corpus.stopwords.words('english')
   # form_str_stop = [word for word in form_str_join.split() if word.lower() not in words_stop]
  #  return form_str_stop


spam_model = joblib.load('naive_model.joblib')
vectorizer = joblib.load('CountVectorizer.joblib')
inp_text = st.text_area('Paste the text to determine whether it is Positive , Negative or Neutral ',height=200)

vectorised_text = vectorizer.transform([inp_text])
pred = ''
# add a placeholder

def spam_predict(inp_text):
    prediction = spam_model.predict(inp_text)
    if prediction == 1:
        pred = 'Positive'
    elif prediction == -1:
        pred = 'Negative'
    else:
        pred = 'Neutral'
    return pred
if st.button('Chech the text'):
    st.write('The text you entered is:',spam_predict(vectorised_text))
st.text('Created by AKHIL S BABU.')
