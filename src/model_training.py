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

def train_model(processed_data_path, model_base_name="sentiment_model", version="v1"):
    data = pd.read_csv(processed_data_path)
    data['sentiment'] = data['sentiment'].map({'Negative': 0, 'Positive': 1})

    if data.empty:
        raise ValueError("No data available after shuffling. Please check the original data size.")

    X = data['clean_text']
    y = data['sentiment']

    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(X)
    X_seq = tokenizer.texts_to_sequences(X)
    X_pad = pad_sequences(X_seq, maxlen=100)

    y_cat = to_categorical(y, num_classes=2)  # Ensure y is correctly one-hot encoded
    X_train, X_test, y_train, y_test = train_test_split(X_pad, y_cat, test_size=0.2, random_state=42)

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("The resulting train or test set is empty. Ensure the original dataset is large enough.")

    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=128))
    model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=128, callbacks=[early_stopping])

    # Generate versioned model path
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/{model_base_name}_{version}_{date}.h5"
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print("Model saved to", model_path)
