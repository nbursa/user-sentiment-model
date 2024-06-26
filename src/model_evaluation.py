import pandas as pd
from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import os

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
    print(f"Evaluation report:\n{report}")
    return report

def get_latest_model_path(model_base_name, version):
    model_dir = "models"
    model_files = [f for f in os.listdir(model_dir) if f.startswith(model_base_name) and version in f]
    if not model_files:
        raise FileNotFoundError(f"No models found for {model_base_name} with version {version}")
    latest_model_file = sorted(model_files, key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)[0]
    return os.path.join(model_dir, latest_model_file)
