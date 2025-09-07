import pandas as pd
import re
import spacy
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, GRU, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. Load Dataset
dataset = load_dataset("tdavidson/hate_speech_offensive")

df = pd.DataFrame(dataset['train'])
df.head()

# Labels -> one-hot encoding
df = df[['class', 'tweet']]
labels = df['class'].tolist()
y = to_categorical(labels, num_classes=3)

# 2. Text Cleaning Functions
def clean_raw_tweet(text):
  text = text.lower()
  text = re.sub(r"http\s+", " ", text)
  text = re.sub(r"@\w+", " ", text)
  text = re.sub(r"#\w+", " ", text)
  text = re.sub(r"[^a-zA-Z\s]", " ", text)
  text = re.sub(r"\s+", " ", text).strip()
  return text

# Load SpaCy English model (only tokenizer + lemmatizer)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "rextcat"])

def clean_with_spacy(text):
  text = clean_raw_tweet(text)
  doc = nlp(text)
  tokens = [tok.lemma_ for tok in doc if not tok.is_stop and tok.is_alpha and len(tok) > 2]

  return " ".join(tokens)

# Apply preprocessing
texts = df['tweet'].astype(str).apply(clean_with_spacy).tolist()

# 3. Tokenization & Padding
MAX_WORDS = 20000  # maximum vocabulary size
MAX_LEN = 100      # maximum sequence length

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>") # <OOV> = Out Of Vocab
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=MAX_LEN)
print(X)

# 4. Train/Validation/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

# 5. Build GRU Model

model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN),
    Bidirectional(GRU(64, return_sequences=False)),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(3, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.build(input_shape=(None, MAX_LEN))
model.summary()

# 6. Compute Class Weights (to handle imbalance)
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

y_integers = np.argmax(y_train, axis=1)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_integers),
    y=y_integers
)

class_weights = dict(enumerate(class_weights))
print(class_weights)

# 7. Callbacks (EarlyStopping + Checkpoint)
checkpoint = ModelCheckpoint(
    filepath = "best_model_word.h5",
    monitor = "val_accuracy",
    save_best_only = True,
    mode ="max",
    verbose = 1
)

early_stopping = EarlyStopping(
    monitor = "val_accuracy",
    patience = 3,
    restore_best_weights = True,
    verbose = 1
)

# 8. Train Model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, early_stopping],
    verbose=1
)

# 9. Evaluate Model
y_pred = model.predict(X_test).argmax(axis=1)
y_true = y_test.argmax(axis=1)

target_names = ["hate speech", "offensive language", "neither"]
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=target_names))

# 10. Plot Accuracy & Loss
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 11. Load Best Model and Test on New Sentences
from tensorflow.keras.models import load_model
best_model = load_model("best_model_word.h5")

sentences = [
    "i love you!",
    "you are look like shit",
    "go away, everybody hate you"
]

seqs = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(seqs, maxlen=MAX_LEN)

preds = best_model.predict(padded)

idx2class = {0: "hate speech", 1: "offensive language", 2: "neither"}

for i, s in enumerate(sentences):
    class_idx = preds[i].argmax()
    print(f"\nSentence: {s}")
    print("Predicted Label:", idx2class[class_idx])
    print("Probabilities:", preds[i])