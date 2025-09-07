# Hate Speech Detection using GRU + SpaCy

This project is a **text classification model** that detects **hate speech, offensive language, and neutral tweets**.  
It uses the **[Davidson Hate Speech dataset](https://huggingface.co/datasets/tdavidson/hate_speech_offensive)** and is implemented with:

- **TensorFlow / Keras** (GRU model)
- **SpaCy** (text preprocessing and lemmatization)
- **Scikit-learn** (class weights, evaluation metrics)
- **Matplotlib** (training visualization)

---

## 🚀 Features
- Cleans raw tweets (removes links, mentions, hashtags, symbols).
- Lemmatizes tokens and removes stopwords using **SpaCy**.
- Tokenizes and pads sequences with **Keras Tokenizer**.
- Trains a **Bidirectional GRU** model with dropout layers.
- Handles **class imbalance** using computed class weights.
- Implements **EarlyStopping** and **ModelCheckpoint** callbacks.
- Provides **classification reports** and **accuracy/loss plots**.
- Allows you to **test custom sentences** with the trained model.

---

## 📊 Example Results
Example classification on test sentences:

```text
Sentence: i love you!
Predicted Label: neither

Sentence: you are look like shit
Predicted Label: offensive language

Sentence: go away, everybody hate you
Predicted Label: hate speech


⚙️ Installation

git clone https://github.com/MiladNlpAi/hate-speech-detection-gru.git
cd hate-speech-detection-gru

pip install -r requirements.txt
python -m spacy download en_core_web_sm
