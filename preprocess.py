import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = " ".join([word for word in text.split() if word not in STOPWORDS])
    return text


def load_and_preprocess_data(true_path, fake_path):

    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)

    true_df['label'] = 1
    fake_df['label'] = 0

    df = pd.concat([true_df, fake_df], axis=0).reset_index(drop=True)

    df['content'] = df['title'].astype(str) + " " + df['text'].astype(str)

    df['content'] = df['content'].apply(clean_text)

    X = df['content'].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    max_len = 200
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

    return X_train_pad, X_test_pad, y_train, y_test, tokenizer, max_len


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, tokenizer, max_len = load_and_preprocess_data(
        "data/True.csv", "data/Fake.csv"
    )

    print("Train Shape:", X_train.shape)
    print("Test Shape:", X_test.shape)
    print("Sample Text:", tokenizer.sequences_to_texts(X_train[:1]))
