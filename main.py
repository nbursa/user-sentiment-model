import argparse
from datetime import datetime
import src.data_processing as dp
import src.model_training as mt
import src.model_evaluation as me

def main(mode=None, version="v1"):
    raw_data_path = 'data/raw/training.1600000.processed.noemoticon.csv'
    processed_data_path = 'data/processed/training_processed.csv'
    model_name = "sentiment_model"
    
    if mode is None or mode == 'process':
        # Step 1: Process the data
        print("Processing data...")
        dp.process_data(raw_data_path, processed_data_path)

    if mode is None or mode == 'train':
        # Step 2: Train the model
        print("Training model...")
        mt.train_model(processed_data_path, model_name, version)

    if mode is None or mode == 'evaluate':
        # Step 3: Evaluate the model
        print("Evaluating model...")
        model_path = me.get_latest_model_path(model_name, version)
        evaluation_result = me.evaluate_model(processed_data_path, model_path)
        print("Model evaluation results:")
        print(evaluation_result)

    if mode not in [None, 'process', 'train', 'evaluate']:
        print("Invalid mode. Please choose 'process', 'train', or 'evaluate'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run data processing, model training, or model evaluation.')
    parser.add_argument('mode', type=str, nargs='?', help="Mode to run: 'process', 'train', or 'evaluate'")
    parser.add_argument('--version', type=str, default='v1', help="Version of the model")
    args = parser.parse_args()
    main(args.mode, args.version)
