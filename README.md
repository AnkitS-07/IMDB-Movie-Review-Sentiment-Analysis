# 🎬 End-to-End GRU: IMDB Movie Review Sentiment Analysis

This project implements an **end-to-end Natural Language Processing (NLP) pipeline** for sentiment analysis of movie reviews using **recurrent neural networks (RNN)** in **PyTorch**. 

The system classifies IMDB movie reviews as **Positive** or **Negative** and includes model training, evaluation, and an interactive **Streamlit web application** for real-time inference.

---

## 📌 Project Overview

The goal of this project is to demonstrate a **transparent and reproducible NLP workflow**, starting from raw text data and ending with a deployable sentiment analysis application.

Key highlights:
- Custom text preprocessing and tokenization (no pretrained tokenizer)
- Vocabulary construction with special tokens (`<PAD>`, `<UNK>`)
- Numerical encoding and sequence padding
- Comparison of **RNN, LSTM, and GRU** architectures
- Final deployment using **Streamlit**
- Honest handling of model confidence and limitations

---

## 🌐 Live Demo  
👉 Try the deployed web app here: [IMDB Movie Review Sentiment Analysis App](https://imdb-movie-review-sentiment-analysis-ankits-07.streamlit.app)


---
## 📂 Dataset

**IMDB Movie Reviews Dataset (50K reviews)**  
- 25,000 positive reviews  
- 25,000 negative reviews  

Dataset link:  [IMDB Movie Review Dataset – 50K Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

Columns:
- `review` – raw movie review text
- `sentiment` – `positive` or `negative`

---

## 📁 Project Structure

```text
imdb-nlp-pytorch/
│
├── 01_nlp_pipeline.ipynb
│   # Text preprocessing, tokenization, vocabulary construction,
│   # exploratory data analysis, and PyTorch DataLoader preparation
│
├── 02_training_models.ipynb
│   # Training and comparison of RNN, LSTM, and GRU architectures
│
├── 03_prediction.ipynb
│   # Model loading and inference examples on unseen text inputs

├── app.py                          # Streamlit web app for interactive sentiment prediction
├── vocab.json                      # Saved vocabulary mapping tokens to numerical IDs
├── best_gru_model.pt               # Trained GRU model weights and configuration 
├── requirements.txt               
└── README.md                      
```

---

## 🧠 NLP Pipeline

The following preprocessing steps are performed:

1. **Text Cleaning**
   - Lowercasing
   - HTML tag removal
   - Punctuation removal
   - Whitespace normalization

2. **Tokenization**
   - Word-level tokenization using simple whitespace splitting

3. **Vocabulary Construction**
   - Custom vocabulary built from the dataset
   - Special tokens:
     - `<PAD>` → padding
     - `<UNK>` → unknown words

4. **Sequence Encoding**
   - Tokens converted to numerical IDs
   - Fixed sequence length (`MAX_LEN = 500`)
   - Padding and truncation applied

5. **Exploratory Data Analysis**
   - Review length distribution
   - Justification of maximum sequence length

---

## 🏗️ Model Architectures

The following recurrent architectures are implemented and compared:

- **Simple RNN** (baseline)
- **LSTM (Long Short-Term Memory)**
- **GRU (Gated Recurrent Unit)**

All models share:
- Embedding layer
- Recurrent layer (RNN / LSTM / GRU)
- Fully connected output layer (2 classes)
- `CrossEntropyLoss` for training

### Final Model
The **GRU-based model** is selected as the final model due to:
- Better handling of long-term dependencies
- Faster convergence compared to LSTM
- Strong validation performance

The final GRU model uses **packed padded sequences** to correctly ignore padding during training and inference.

---

## 📊 Results Summary

- **Simple RNN** struggles with long reviews and performs close to random guessing
- **LSTM** significantly improves performance
- **GRU** achieves comparable or better accuracy with fewer parameters

The final GRU model generalizes well to full-length reviews.

---

## ⚠️ Model Limitations

- The model is trained on **full movie reviews**
- Very short inputs (e.g., single sentences) may be misclassified
- Confidence scores are probabilistic, not guarantees

To address this:
- The app displays a warning for very short inputs
- Confidence interpretation text is shown to guide users

---

## 🔮 Future Improvements

The project can be further extended in several directions:

- Incorporating **Bidirectional GRU/LSTM** for improved contextual understanding
- Using **pretrained word embeddings** such as GloVe or FastText
- Applying **transformer-based models** (e.g., BERT) for higher accuracy
- Training with **sentence-level sentiment datasets** to improve short-text predictions
- Adding **model explainability** techniques (e.g., attention visualization)
- Deploying the app using **Docker or cloud platforms**

---

## ✅ Conclusion

This project presents a complete **end-to-end NLP sentiment analysis system** built using **PyTorch** and recurrent neural networks.  
Starting from raw IMDB reviews, the pipeline covers text preprocessing, custom vocabulary construction, sequence modeling, and deployment through a Streamlit application.

By comparing **RNN, LSTM, and GRU** architectures, the project highlights the strengths and limitations of each model and justifies the final selection of a **GRU-based model**.  
Special care is taken to handle variable-length inputs, model confidence, and real-world limitations, making the system both **technically sound and practically honest**.

---

## 📄 License

Standard MIT License

---

**Author:** Ankit Sarkar  
**Project:** End-to-End GRU based IMDB Movie Review Sentiment Analysis  
**Date:** January 2026

---
