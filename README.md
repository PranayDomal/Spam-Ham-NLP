# **Spam vs Ham Classification using NLP & Naive Bayes**

## **Project Overview**

This project implements a **text-based spam detection system** that classifies SMS messages as **Spam** or **Ham** using classical **Natural Language Processing (NLP)** techniques and a Multinomial Naive Bayes classifier.
The focus of the project is not only on building a classifier, but on correct evaluation, handling class imbalance, and avoiding data leakage.

## **Dataset Description**

- Source: SMS Spam Collection Dataset

- Total Messages: 5,574

- Classes:

  - Ham: ~86.6%

  - Spam: ~13.4%

- Target Variable: type (spam / ham)

- Feature: Raw SMS text

The dataset is **heavily imbalanced**, making spam recall a more meaningful metric than overall accuracy.

## **Exploratory Data Analysis (EDA)**

Key observations:

- Spam messages tend to be longer than ham messages on average

- A small number of very long messages exist as outliers

- Text data contains noise such as punctuation, stopwords, and informal language

EDA was used to understand message length distribution and class behavior, not to engineer leakage-prone features.

## **Text Preprocessing**

The following preprocessing steps were applied:

- Removal of punctuation

- Tokenization

- Stopword removal using NLTK

- No modification of raw text during EDA (preprocessing applied only during vectorization)

A custom analyzer function was used inside the vectorizer to ensure consistency.

## **Feature Engineering**

Two-stage feature extraction was used:

1. **Bag of Words (BoW)**

    - Implemented using `CountVectorizer`

    - Vocabulary learned only from training data

2. **TF-IDF Transformation**

    - Applied using TfidfTransformer

    - Prevents common words from dominating the feature space

This approach keeps the model simple, interpretable, and efficient.

## **Modeling Approach**

- Algorithm: Multinomial Naive Bayes

- Smoothing Parameter (`alpha`): 0.5

- Train/Test Split: 80/20 (stratified)

A stratified split was introduced before vectorization to eliminate data leakage and ensure realistic evaluation.

## **Model Evaluation**

Evaluation was performed only on the test set using:

- Precision

- Recall

- F1-score

- Confusion Matrix

**Test Set Performance (Approximate)**

| Class | Precision | Recall | F1-score |
| ----- | --------- | ------ | -------- |
| Ham   | ~0.97     | ~1.00  | ~0.99    |
| Spam  | ~0.99     | ~0.82  | ~0.90    |


## **Tools & Libraries**

- Python

- Pandas, NumPy

- Matplotlib, Seaborn

- NLTK

- scikit-learn

## **How to Run**

1. Clone the repository:
```bash
git clone https://github.com/PranayDomal/Spam-Ham-NLP.git
```

2. Navigate to the folder:
```bash
cd Spam-Ham-NLP
```

3. Run the notebook:
```bash
jupyter notebook spam_ham_NLP.ipynb
```

## **Author**

https://www.linkedin.com/in/pranay-domal-a641bb368/
