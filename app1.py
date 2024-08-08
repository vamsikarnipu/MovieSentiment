import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load your model here
from tensorflow.keras.models import load_model
model = load_model('MovieSentiment.keras')  # Adjust the path to your model file

# Initialize NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer
lm = WordNetLemmatizer()

def preprocess_review(review):
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [lm.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    review = one_hot(review, 25000)
    review = pad_sequences([review], padding='pre', maxlen=30)
    return review

def predict_sentiment(review):
    preprocessed_review = preprocess_review(review)
    output = model.predict(preprocessed_review)
    score = output[0][0]
    return score

# Streamlit app
st.title('Movie Review Sentiment Analysis')
st.write("Enter your movie review below:")

# Text input for the review
review_input = st.text_area("Review")

if st.button('Analyze'):
    if review_input:
        score = predict_sentiment(review_input)
        if score > 0.5:
            st.write(f"Score: {score:.2f} - Positive")
        else:
            st.write(f"Score: {score:.2f} - Negative")
    else:
        st.write("Please enter a review to analyze.")
