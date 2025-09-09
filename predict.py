import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import clean_text

model = tf.keras.models.load_model("models/fake_news_model.h5", compile=False)
tokenizer = joblib.load("models/tokenizer.pkl")

MAX_LEN = 500

def predict_news(title, text):
    content = str(title) + " " + str(text)
    
    cleaned = clean_text(content)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    prob = model.predict(padded)[0][0]
    label = "Real News" if prob >= 0.5 else "Fake News"

    return label, float(prob)
if __name__ == "__main__":
    title = "Breaking: Aliens Land in New York City"
    text = "Witnesses report a UFO hovering above Times Square. The government confirms contact with extraterrestrials."
    
    label, prob = predict_news(title, text)
    print(f"Prediction: {label} (Confidence: {prob:.4f})")

    title = "Phillipson and Thornberry among six Labour deputy hopefuls"
    text = "Education Secretary Bridget Phillipson and senior backbencher Dame Emily Thornberry are among six Labour MPs to have entered the contest to be party's next deputy leader.Housing Minister Alison McGovern, former minister Lucy Powell, and backbenchers Paula Barker and Bell Ribeiro-Addy are also running in the race to replace Angela Rayner in the deputy role.In order to run, the candidates will need to collect support from 80 MPs by 17:00 on Thursday - a tight timetable which has been criticised by some in Labour.The contest will impact the future direction of Labour, as the party grapples with political competition from the left and right."
    
    label, prob = predict_news(title, text)
    print(f"Prediction: {label} (Confidence: {prob:.4f})")
