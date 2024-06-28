import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Embedding, LSTM, Dense # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from datetime import datetime
import matplotlib.pyplot as plt
import joblib


def train_model(processed_data_path, model_base_name="sentiment_model", version="",
                embedding_dim=128, lstm_units=128, dropout_rate=0.2, epochs=1, batch_size=128):

    recurrent_dropout_rate = 0.2
    max_words = 10000
    max_len = 100

    # Load the processed data
    print(f"Loading data from {processed_data_path}")
    data = pd.read_csv(processed_data_path)
    print(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")

    # Use a subset of the data for faster training
    data = data.sample(frac=0.1, random_state=42)
    print(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")

    # Check if the dataset is empty
    if data.empty:
        raise ValueError("No data available. Please check the processed data.")

    X = data['clean_text']
    y = data['sentiment']

    print("Tokenizing and padding text data...")
    # Tokenize and pad the text data
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X)
    X_seq = tokenizer.texts_to_sequences(X)
    X_pad = pad_sequences(X_seq, maxlen=max_len)
    print(f"Tokenization and padding completed. Shape of X_pad: {X_pad.shape}")

    # Save the tokenizer
    date = datetime.now().strftime("%d%m%Y%H%M%S")
    tokenizer_path = f'models/{model_base_name}_tokenizer_{version}_{date}.joblib'
    print("Saving tokenizer to", tokenizer_path)
    joblib.dump(tokenizer, tokenizer_path)

    print("Converting sentiment labels to categorical...")
    # Convert 'y' to categorical
    y_cat = to_categorical(y, num_classes=2)
    print(f"Conversion to categorical completed. Shape of y_cat: {y_cat.shape}")

    print("Splitting data into training and test sets...")
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_pad, y_cat, test_size=0.2, random_state=42)
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

    # Ensure training and test sets are not empty
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("The resulting train or test set is empty. Ensure the original dataset is large enough.")

    print("Building the model...")
    # Build the model
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=embedding_dim))
    model.add(LSTM(units=lstm_units, dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate))
    model.add(Dense(2, activation='softmax'))

    print("Compiling the model...")
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    print("Training the model...")
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])
    print("Model training completed.")

    # Generate versioned model path
    model_path = f"models/{model_base_name}_{version}_{date}_embedding_dim={embedding_dim}_lstm_units={lstm_units}_dropout_rate={dropout_rate}_epochs={epochs}_batch_size={batch_size}.keras"

    print(f"Saving the model to {model_path}...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print("Model saved to", model_path)

    # Save the training history
    history_path = f'models/{model_base_name}_history_{version}_{date}.csv'
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(history_path, index=False)
    print("Training history saved to", history_path)

    return history, model_path, tokenizer_path


def plot_training_history(history, model_base_name, version, date):
    # Plot the training history
    if not history.history:
        raise ValueError("The training history is empty. Ensure that the model training process was successful.")

    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    accuracy_plot_path = f'images/{model_base_name}_accuracy_{version}_{date}.png'
    plt.savefig(accuracy_plot_path)
    plt.show()
    print("Accuracy plot saved to", accuracy_plot_path)

    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    loss_plot_path = f'images/{model_base_name}_loss_{version}_{date}.png'
    plt.savefig(loss_plot_path)
    plt.show()
    print("Loss plot saved to", loss_plot_path)
