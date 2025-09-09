import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from preprocess import load_and_preprocess_data

X_train, X_test, y_train, y_test, tokenizer, max_len = load_and_preprocess_data(
    "data/True.csv", "data/Fake.csv"
)

model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=max_len),
    LSTM(128, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_test, y_test)
)
loss, acc = model.evaluate(X_test, y_test)
model.save("models/fake_news_model.h5")
import joblib
joblib.dump(tokenizer, "models/tokenizer.pkl")
