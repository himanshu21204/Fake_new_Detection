import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from preprocess import load_and_preprocess_data
import joblib
import os

# ----------------------------
# Load and preprocess data
# ----------------------------
X_train, X_test, y_train, y_test, tokenizer, max_len = load_and_preprocess_data(
    "data/True.csv", "data/Fake.csv"
)

# ----------------------------
# Build improved LSTM model
# ----------------------------
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=max_len),
    LSTM(128, return_sequences=True, dropout=0.3),  # Removed recurrent_dropout for Python 3.12
    BatchNormalization(),
    LSTM(64, dropout=0.3),
    Dropout(0.4),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

# Compile model
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# ----------------------------
# Callbacks
# ----------------------------
if not os.path.exists("models"):
    os.makedirs("models")

callbacks = [
    EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ModelCheckpoint("models/fake_news_model.h5", save_best_only=True, monitor="val_loss")
]

# ----------------------------
# Train model
# ----------------------------
history = model.fit(
    X_train, y_train,
    epochs=20,               # Increased epochs for better learning
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {acc*100:.2f}%")

# Save tokenizer
joblib.dump(tokenizer, "models/tokenizer.pkl")
print("✅ Model and tokenizer saved!")
