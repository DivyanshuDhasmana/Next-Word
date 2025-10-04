import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# ----------------- Load Model & Tokenizer -----------------
model = load_model("my_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Build reverse mapping (index → word)
index_word = {index: word for word, index in tokenizer.word_index.items()}

# ----------------- Prediction Function -----------------
def generate_text(seed_text, next_words=10):
    text = seed_text
    for _ in range(next_words):
        # tokenize
        token_text = tokenizer.texts_to_sequences([text])[0]
        # padding
        padded_token_text = pad_sequences([token_text], maxlen=56, padding='pre')
        # predict
        predictions = model.predict(padded_token_text, verbose=0)
        predicted_index = np.argmax(predictions)
        # decode word
        predicted_word = index_word.get(predicted_index, "")
        # append
        text += " " + predicted_word
    return text

# ----------------- Streamlit UI -----------------
st.title("🔮 Next Word Prediction App")
st.write("Type a sentence and the model will generate the next words.")

# User input
seed_text = st.text_input("Enter your starting sentence:", "what is the fee")
num_words = st.slider("How many words to generate?", 1, 20, 10)

if st.button("Generate"):
    result = generate_text(seed_text, next_words=num_words)
    st.success(f"Generated Text:\n\n{result}")
