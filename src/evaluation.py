import pandas as pd
from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import os
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(processed_data_path, model_path, batch_size):
    print(f"Loading data from {processed_data_path}...")
    data = pd.read_csv(processed_data_path)

    # Debugging: Print first few rows and unique values in 'sentiment' column
    print("First few rows of the processed data:")
    print(data.head())
    print(f"Unique sentiment values: {data['sentiment'].unique()}")

    X = data['clean_text']
    y = data['sentiment']

    print("Initializing and fitting tokenizer...")
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(X)

    print("Converting texts to sequences...")
    X_seq = tokenizer.texts_to_sequences(X)

    print("Padding sequences...")
    X_pad = pad_sequences(X_seq, maxlen=100)

    print(f"Loading model from {model_path}...")
    model = load_model(model_path)

    print("Predicting in batches to avoid memory issues...")
    y_pred = model.predict(X_pad, batch_size=batch_size)
    y_pred_classes = y_pred.argmax(axis=1)

    print("Generating classification report...")
    report = classification_report(y, y_pred_classes, target_names=['Negative', 'Positive'])
    print("Classification Report:\n", report)

    print("Generating confusion matrix...")
    cm = confusion_matrix(y, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    print("Saving confusion matrix plot...")
    plt.savefig('images/confusion_matrix.png')
    plt.show()

    return report


def get_latest_model_path(model_base_name):
    print(f"Searching for latest model with base name {model_base_name} in 'models' directory...")
    model_dir = "models"
    model_files = [f for f in os.listdir(model_dir) if f.startswith(model_base_name)]
    if not model_files:
        raise FileNotFoundError(f"No models found for {model_base_name}")

    print("Sorting model files by modification time...")
    latest_model_file = sorted(model_files, key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)[0]
    latest_model_path = os.path.join(model_dir, latest_model_file)
    print(f"Latest model path: {latest_model_path}")

    return latest_model_path
