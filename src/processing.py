import pandas as pd
import os
import re

def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove special characters and numbers
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def process_data(raw_data_path, processed_data_path):
    print(f"Loading data from {raw_data_path}...")
    # Load data and assign column names
    data = pd.read_csv(raw_data_path, header=None, encoding='latin-1')
    data.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
    print("First few rows of the raw data:")
    print(data.head())

    print("Dropping 'query' column...")
    # Drop the 'query' column as it's not needed
    data.drop(columns=['query'], inplace=True)

    print("Removing timezone information from the 'date' column...")
    # Remove timezone information from the 'date' column
    data['date'] = data['date'].str.replace(r' [A-Z]{3}', '', regex=True)

    print("Converting 'date' column to datetime format...")
    # Convert 'date' column to datetime
    data['date'] = pd.to_datetime(data['date'], errors='coerce')

    print("Cleaning the text data...")
    # Clean the text data
    data['clean_text'] = data['text'].apply(clean_text)
    print("First few rows of data after cleaning text:")
    print(data[['clean_text']].head())

    print("Mapping sentiment values to 0 (Negative) and 1 (Positive)...")
    # Map sentiment values to 0 (Negative) and 1 (Positive)
    data['sentiment'] = data['sentiment'].map({0: 'Negative', 4: 'Positive'})
    data['sentiment'] = data['sentiment'].map({'Negative': 0, 'Positive': 1})

    print("Dropping rows with NaN values in 'clean_text' or 'sentiment'...")
    # Drop rows with NaN values in 'clean_text' or 'sentiment'
    data.dropna(subset=['clean_text', 'sentiment'], inplace=True)

    print("Dropping duplicates...")
    # Drop duplicates
    data = data.drop_duplicates()

    print("Dropping the original 'text' column...")
    # Drop the original 'text' column as it's no longer needed
    data.drop(columns=['text'], inplace=True)

    print("First few rows of the processed data:")
    print(data.head())

    print("Shuffling the data...")
    # Shuffle the data
    data = data.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print("Saving processed data to CSV file...")
    # Save processed data to a CSV file
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    data.to_csv(processed_data_path, index=False)
    print(f"Processed data saved to {processed_data_path}")
