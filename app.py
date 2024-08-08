from flask import Flask, request, render_template, jsonify
import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
lm=WordNetLemmatizer()

app = Flask(__name__)

# Initialize lemmatizer and stopwords
lm = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load the pre-trained model
model = load_model('MovieSentiment.keras')  # Make sure you have this model file


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    Review=review
    review=re.sub('[^a-zA-Z]',' ',review)
    review=review.lower()
    review=review.split()
    review=[lm.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review=' '.join(review)
    print(review)
    review=one_hot(review,25000)
    review=pad_sequences([review],padding='pre',maxlen=30)
    output=model.predict(review)
    prediction=np.where(output>0.5,1,0)
    sentiment = 'positive' if prediction > 0.5 else 'negative'
    return render_template('index.html', review=Review, sentiment=sentiment, score=float(output))

if __name__ == '__main__':
    app.run(debug=True)
