import streamlit as st
import torch
import torch.nn as nn
import json
import re
import numpy as np
import os
import gdown


# Download model from Google Drive if not present
MODEL_PATH = "best_gru_model.pt"
GOOGLE_DRIVE_URL = "https://drive.google.com/uc?id=1cFKISTZsuBGyUsIEUjVTvp5IhBbqDlgk"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading trained model..."):
        gdown.download(GOOGLE_DRIVE_URL, MODEL_PATH, quiet=False)


@st.cache_resource
def load_model_and_vocab():
    checkpoint = torch.load("best_gru_model.pt", map_location="cpu")

    vocab = checkpoint["vocab"]
    embed_dim = checkpoint["embed_dim"]
    hidden_dim = checkpoint["hidden_dim"]

    vocab_size = len(vocab)

    class SentimentGRU(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_dim):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.recurrent = nn.GRU(embed_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 2)

        def forward(self, x, lengths):
            x = self.embedding(x)

            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

            _, hidden = self.recurrent(packed)
            return self.fc(hidden[-1])

    model = SentimentGRU(vocab_size, embed_dim, hidden_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, vocab


MAX_LEN = 500


def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    return text.split()


def is_short_input(tokens, min_tokens=5):
    return len(tokens) < min_tokens


def encode_and_pad(tokens, vocab, max_len):
    encoded = [vocab.get(word, vocab["<UNK>"]) for word in tokens]
    length = min(len(encoded), max_len)

    if len(encoded) < max_len:
        encoded += [vocab["<PAD>"]] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]

    return encoded, length


def predict_sentiment(text, model, vocab):
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)

    if len(tokens) == 0:
        return None, None

    encoded, length = encode_and_pad(tokens, vocab, MAX_LEN)

    x = torch.tensor([encoded], dtype=torch.long)
    lengths = torch.tensor([length])

    with torch.no_grad():
        outputs = model(x, lengths)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    sentiment = "Positive 😊" if pred.item() == 1 else "Negative 😞"
    return sentiment, confidence.item()


def interpret_confidence(confidence):
    if confidence >= 0.85:
        return "🟢 The model is very confident in this prediction."
    elif confidence >= 0.65:
        return "🟡 The model is moderately confident in this prediction."
    else:
        return "🔴 The model is uncertain. Consider providing a longer or more detailed review."


st.set_page_config(page_title="IMDB Sentiment Analyzer", layout="centered")

st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write(
    "This application demonstrates an end-to-end **NLP pipeline** using a "
    "**GRU-based neural network** trained on the IMDB movie review dataset "
    "to predict sentiment as **Positive** or **Negative**."
)

model, vocab = load_model_and_vocab()

user_input = st.text_area(
    "✍️ Enter a movie review:",
    height=150,
    placeholder="Type your movie review here..."
)

if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a movie review.")
    else:
        cleaned = clean_text(user_input)
        tokens = tokenize(cleaned)

        if is_short_input(tokens):
            st.info(
                "ℹ️ This model is trained on full movie reviews. "
                "Predictions for very short inputs may be less reliable."
            )

        sentiment, confidence = predict_sentiment(user_input, model, vocab)

        if sentiment is None:
            st.error("Unable to process the input text.")
        else:
            st.subheader("📊 Prediction Result")
            st.write(f"**Sentiment:** {sentiment}")
            st.write(f"**Confidence:** {confidence:.3f}")
            st.write(interpret_confidence(confidence))
            st.progress(min(confidence, 1.0))


st.markdown(
    "<p style='text-align: center; font-size:14px;'> Made with 💡 by <b>Ankit Sarkar</b></p>",
    unsafe_allow_html=True
)
