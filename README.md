# Spam/Ham Message Classifier

## Description:
This project implements a simple yet effective text classification model to categorize messages as either "spam" or "ham" (legitimate, non-spam). It utilizes a TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer to convert text data into numerical features, followed by a Multinomial Naive Bayes classifier. This approach is a popular and robust baseline for tasks such as email and SMS filtering.

## Features:
- **TF-IDF Vectorization**: Transforms text messages into numerical feature vectors, capturing the importance of words within the messages.
- **Multinomial Naive Bayes Classifier**: A probabilistic classifier well-suited for text classification tasks.
- **Pipeline Integration**: Uses `sklearn.pipeline.make_pipeline` for a streamlined workflow from text preprocessing to classification.
- **Model Evaluation**: Includes `classification_report` to assess the model's performance on test data.
- **Prediction on New Messages**: Demonstrates how to use the trained model to classify new, unseen messages.

## Getting Started:

### Prerequisites
This project requires Python and the following libraries:
- `scikit-learn`

### Installation
```bash
pip install scikit-learn
```

### Usage
1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Run the Jupyter/Colab notebook:**
    Open the provided notebook (e.g., `spam_ham_classifier.ipynb`) in a Jupyter environment or Google Colab.

3.  **Execute the cells:**
    The notebook contains cells for:
    -   Loading and preparing a sample dataset.
    -   Training the TF-IDF + Naive Bayes model.
    -   Evaluating the model's performance.
    -   Testing the model with new messages.

## Project Structure (if applicable):
-   `spam_ham_classifier.ipynb`: The main Jupyter/Colab notebook containing the code.
-   `data/`: (Optional) Directory for storing datasets.

## Dataset:
The model is demonstrated with a small, hand-crafted sample dataset of messages and their corresponding "spam" or "ham" labels. For real-world applications, a larger and more diverse dataset would be required.

## Evaluation Metrics:
The model's performance is evaluated using `classification_report`, which provides:
-   **Precision**: The proportion of positive identifications that were actually correct.
-   **Recall**: The proportion of actual positives that were identified correctly.
-   **F1-score**: The harmonic mean of precision and recall.
-   **Support**: The number of actual occurrences of each class in the specified dataset.

## Future Enhancements:
-   Integrate a larger, real-world dataset for training and evaluation.
-   Explore other vectorization techniques (e.g., Word2Vec, GloVe, BERT embeddings).
-   Experiment with different classification algorithms (e.g., SVM, Logistic Regression, deep learning models).
-   Implement hyperparameter tuning for improved model performance.
-   Add cross-validation for more robust evaluation.
