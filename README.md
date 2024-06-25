# User Sentiment Model

This project is designed to analyze the sentiment of user tweets, aiming to support a broader analysis of user motivations on Twitter. The project includes data preprocessing, model training, and model evaluation using an LSTM (Long Short-Term Memory) model.

## Project Structure

```
user-sentiment-model
├── .venv
├── data
│   ├── processed
│   │   ├── most-active-users.png
│   │   ├── sentiment-distribution.png
│   │   ├── sentiment-time-distribution.png
│   │   ├── training_processed.csv
│   ├── raw
│   │   └── training.1600000.processed.noemoticon.csv
├── models
├── notebooks
│   └── data_exploration.ipynb
├── src
│   ├── data_processing.py
│   ├── model_evaluation.py
│   ├── model_training.py
│   └── main.py
├── requirements.txt
└── setup.py
```

## Setup

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

- Included data set file (`training.1600000.processed.noemoticon.csv`) obtained from  in the `data/raw` directory.
- Processed data, models, and visualizations will be saved in their respective directories during the pipeline execution.

## Usage

### Data Processing

1. **Run data processing:**
    ```sh
    python src/data_processing.py
    ```

### Model Training

2. **Train the model:**
    ```sh
    python src/model_training.py
    ```

### Model Evaluation

3. **Evaluate the model:**
    ```sh
    python src/model_evaluation.py
    ```

### Main Pipeline

You can also run the entire pipeline with the `main.py` script:
```sh
python src/main.py
```

## Project Files

### `src/data_processing.py`

This script handles the preprocessing of raw tweet data. It includes tasks such as removing unnecessary columns, cleaning text data, and saving the processed data to a CSV file.

### `src/model_training.py`

This script trains an LSTM model on the processed tweet data. It includes tokenization, padding sequences, and saving the trained model.

### `src/model_evaluation.py`

This script evaluates the trained model using various performance metrics and generates visualizations.

### `notebooks/data_exploration.ipynb`

This Jupyter notebook contains exploratory data analysis (EDA) to understand the distribution of the data, identify potential issues, and visualize key insights.

## Key Components

- **LSTM Model:** Used for its ability to handle the vanishing gradient problem and effectively understand context in sequences, such as tweets.
- **Sentiment Analysis:** Determines the sentiment of user tweets (positive or negative).
- **Data Visualization:** Provides insights into the sentiment distribution, most active users, and sentiment over time.

## Frameworks and Libraries

- TensorFlow and Keras for the deep learning framework.
- Scikit-learn for data processing and model evaluation tools.
- Pandas for data manipulation.
- Matplotlib and Seaborn for data visualization.

---