import streamlit as st
import re
from collections import Counter
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
import nltk #nltk --> Natural Language Toolkit
stemmer = nltk.PorterStemmer() # Stemming each word i.e., getting the root for each word in an email
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

model = joblib.load(open('App/new_model.joblib', 'rb'))
with open('App/reference_words.joblib', 'rb') as f:
    ref_words = joblib.load(f)

def stemming_word(text, lower_case=True):
    if lower_case:
        text = text.lower() #Converting all the cases in the test to lower case
    text = re.sub(r'\d+', 'Number', text) #Using RegEx to change all the numbers in the text to the string 'NUMBER'
    text = re.sub(r'http\S+|www.\S+', 'URL', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    pattern = r'^\s*_+\s*$'
    text = re.sub(pattern, '', text, flags=re.MULTILINE)
    text = re.sub(r'\b\w*nonumber\w*\b', '', text)
    word_counts = Counter(text.split()) 
    #  Split the text by the spaces between each word and count the frequency of each word then, 
    #  store the word and its count in a dictionary (That is what Counter() does).

    stemmed_word_counts = Counter()
    for word, count in word_counts.items():
        stemmed_word = stemmer.stem(word) #This give the root of each word using nltk library
        stemmed_word_counts[stemmed_word] += count #Frequency of the stemmed word
    word_counts = stemmed_word_counts
    return word_counts

class Clean_Text(BaseEstimator, TransformerMixin):
    def __init__(self):
        self
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_transformed = []
        X_transformed.append(stemming_word(X))
        return np.array(X_transformed)
    
class Clean_df(BaseEstimator, TransformerMixin):
    def __init__(self):
        self
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        text_df = pd.DataFrame(columns=ref_words)
        i = 0
        row = [None] * len(ref_words)
        for k, v in X[0].items():
                if k in ref_words:
                    row[ref_words.index(k)] = v
                text_df.loc[i] = row
        df = text_df.fillna(0)
        return df
    
from sklearn.pipeline import Pipeline
preprocess_pipeline = Pipeline([
    ("email_to_wordcount", Clean_Text()),
    ("wordcount_to_vector", Clean_df())
])

def main():
    st.title('Spam Email Classifier')
    
    # Input variables
    with st.form('Input form'):
        email_content = st.text_area('Paste your email content here:', height=300)
        submit = st.form_submit_button('Predict')

    # Check if anything is inputed
    if submit:

        transformed_text = preprocess_pipeline.fit_transform(email_content)
        prediction = model.predict(transformed_text)
            
        if prediction == 0:
            st.write('This email is not spam.')

        else:
            st.write('This email is spam.')

if __name__ == '__main__':
    main()
    
