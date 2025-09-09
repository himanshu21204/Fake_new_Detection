from flask import Flask, render_template, request
import numpy as np
import joblib
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import clean_text 

app = Flask(__name__)

MODEL_PATH = "models/fake_news_model.h5"
TOKENIZER_PATH = "models/tokenizer.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
tokenizer = joblib.load(TOKENIZER_PATH)
max_len = 200

def preprocess_text(text):
    return clean_text(text)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        title = request.form['title']
        text = request.form['text']

        combined_text = title + " " + text
        processed_text = preprocess_text(combined_text)

        seq = tokenizer.texts_to_sequences([processed_text])
        pad = pad_sequences(seq, maxlen=max_len, padding="post")

        pred = model.predict(pad)[0][0]
        label = "Real News" if pred >= 0.5 else "Fake News"
        confidence = pred if pred >= 0.5 else 1 - pred

        return render_template('result.html',
                               title=title,
                               text=text,
                               label=label,
                               confidence=f"{confidence*100:.2f}%")

if __name__ == '__main__':
    app.run(debug=True)
