import pandas as pd
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def evaluate_model(processed_data_path, model_path):
    data = pd.read_csv(processed_data_path)
    data['sentiment'] = data['sentiment'].map({'Negative': 0, 'Positive': 1})
    X = data['clean_text']
    y = data['sentiment']
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(X)
    X_seq = tokenizer.texts_to_sequences(X)
    X_pad = pad_sequences(X_seq, maxlen=100)
    model = load_model(model_path)
    y_pred = model.predict(X_pad)
    y_pred_classes = y_pred.argmax(axis=1)
    report = classification_report(y, y_pred_classes, target_names=['Negative', 'Positive'])
    return report
