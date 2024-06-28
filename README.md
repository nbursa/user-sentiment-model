# User Sentiment Model

Project goal is to train a machine learning model designed to analyze the sentiment of user tweets, aiming to support a broader analysis of user motivations on Twitter. The project includes data preprocessing, model training, and model evaluation using an LSTM (Long Short-Term Memory) model.

## Setup

⚠️ Warning: This project is in progress and may contain bugs or incomplete features.

### Prerequisites

- Python 3.x
- Virtual environment tool (venv, virtualenv, etc.)
- PyCharm (optional, but recommended for ease of use)

### Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/nbursa/user-sentiment-model.git
    cd user-sentiment-model
    ```

2. **Create and activate a virtual environment:**
    ```sh
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3. **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

### Directory Setup

- Include dataset file in the `data/raw` directory.
- Processed data, models, and visualizations will be saved in their respective directories during the pipeline execution.

---

## Usage

### Data Processing

1. **Run data processing:**
    ```sh
    python main.py process
    ```

### Model Training

2. **Train the model with specific parameters:**
    ```sh
    python main.py train --version <version> --embedding_dim <embedding_dim> --lstm_units <lstm_units> --dropout_rate <dropout_rate> --epochs <epochs> --batch_size <batch_size>
    ```

   - Example:
     ```sh
     python main.py train --version v1 --embedding_dim 128 --lstm_units 128 --dropout_rate 0.2 --epochs 1 --batch_size 128
     ```

### Model Evaluation

3. **Evaluate the model:**
    ```sh
    python main.py eval --batch_size <batch_size>
    ```

   - Example:
     ```sh
     python main.py eval --batch_size 2000
     ```

---

## Project Files

### `src/main.py`

This script handles the main pipeline, including data processing, model training, and model evaluation.

### `src/data_processing.py`

This script handles the preprocessing of raw tweet data. It includes tasks such as removing unnecessary columns, cleaning text data, and saving the processed data to a CSV file.

### `src/model_training.py`

This script trains an LSTM model on the processed tweet data. It includes tokenization, padding sequences, and saving the trained model.

### `src/model_evaluation.py`

This script evaluates the trained model using various performance metrics and generates visualizations.

### `notebooks/data_exploration.ipynb`

This Jupyter notebook contains exploratory data analysis (EDA) to understand the distribution of the data, identify potential issues, and visualize key insights.

---

## Key Components

- **LSTM Model:** Used for its ability to handle the vanishing gradient problem and effectively understand context in sequences, such as tweets.
- **Sentiment Analysis:** Determines the sentiment of user tweets (positive or negative).
- **Data Visualization:** Provides insights into the sentiment distribution, most active users, and sentiment over time.

---

## Frameworks and Libraries

- TensorFlow and Keras for the deep learning framework.
- Scikit-learn for data processing and model evaluation tools.
- Pandas for data manipulation.
- Matplotlib and Seaborn for data visualization.

---