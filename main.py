import src.data_processing as dp
import src.model_training as mt
import src.model_evaluation as me


def main():
    # Step 1: Process the data
    print("Processing data...")
    raw_data_path = 'data/raw/training.1600000.processed.noemoticon.csv'
    processed_data_path = 'data/processed/training_processed.csv'
    dp.process_data(raw_data_path, processed_data_path)
    print("Data processed and saved to", processed_data_path)

    # Step 2: Train the model
    print("Training model...")
    model_path = 'models/model.h5'
    mt.train_model(processed_data_path, model_path)
    print("Model trained and saved to", model_path)

    # Step 3: Evaluate the model
    print("Evaluating model...")
    evaluation_result = me.evaluate_model(processed_data_path, model_path)
    print("Model evaluation results:")
    print(evaluation_result)


if __name__ == '__main__':
    main()
