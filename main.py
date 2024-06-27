import argparse
import src.processing as processing
import src.training as training
import src.evaluation as evaluation
import matplotlib.pyplot as plt

def main(mode=None, version="v0", embedding_dim=128, lstm_units=128, dropout_rate=0.2, epochs=1, batch_size=128):
    raw_data_path = 'data/raw/training.1600000.processed.noemoticon.csv'
    processed_data_path = 'data/processed/training_processed.csv'
    model_name = "sentiment_model"
    
    if mode == 'process':
        # Step 1: Process the data
        print("Processing data...")
        processing.process_data(raw_data_path, processed_data_path)

    elif mode == 'train':
        # Step 2: Train the model
        print("Training model...")
        history = training.train_model(processed_data_path, model_name, version, embedding_dim, lstm_units, dropout_rate, epochs, batch_size)
        plot_training_history(history)

    elif mode == 'eval':
        # Step 3: Evaluate the model
        print("Evaluating model...")
        model_path = evaluation.get_latest_model_path(model_name)
        evaluation_result = evaluation.evaluate_model(processed_data_path, model_path, batch_size=batch_size)
        print("Model evaluation results:")
        print(evaluation_result)

    else:
        print("Invalid mode. Please choose 'process', 'train', or 'eval'.")

def plot_training_history(history):
    # Summarize history for accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Summarize history for loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('images/training_history.png')
    
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run data processing, model training, or model evaluation.')
    parser.add_argument('mode', type=str, help="Mode to run: 'process', 'train', or 'eval'")
    parser.add_argument('--version', type=str, default='v0', help="Version of the model")
    parser.add_argument('--embedding_dim', type=int, default=128, help="Dimension of the embedding layer")
    parser.add_argument('--lstm_units', type=int, default=128, help="Number of LSTM units")
    parser.add_argument('--dropout_rate', type=float, default=0.2, help="Dropout rate for the LSTM layer")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training and evaluation")
    args = parser.parse_args()
    
    # Validate dropout rate only if mode is 'train'
    if args.mode == 'train' and not (0 <= args.dropout_rate < 1):
        raise ValueError("Dropout rate must be a float in the range [0, 1).")
    
    main(args.mode, args.version, args.embedding_dim, args.lstm_units, args.dropout_rate, args.epochs, args.batch_size)
