import pandas as pd
import os
import re


def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def process_data(raw_data_path, processed_data_path):
    data = pd.read_csv(raw_data_path, header=None)
    data.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
    data.drop(columns=['query'], inplace=True)
    data['date'] = data['date'].astype(str)
    data['date'] = data['date'].str.replace(r' [A-Z]{3}', '', regex=True)
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data = data.drop_duplicates()
    data['clean_text'] = data['text'].apply(clean_text)
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    data.to_csv(processed_data_path, index=False)
    print(f"Processed data saved to {processed_data_path}")
