# AI Detector

## 🚀 Project Overview
AI Detector is a machine learning and deep learning-based system designed to classify whether a given text is human-written or AI-generated. The project leverages **Natural Language Processing (NLP)** techniques, **feature engineering**, and **pre-trained transformer models** to analyze textual patterns and make accurate predictions.

## 🔍 Problem Statement
With the increasing presence of AI-generated content, distinguishing between human-written and AI-generated text is crucial for journalism, academic integrity, and content moderation. This project aims to build an accurate classifier for this task.

## 📌 Features & Approach
- **Data Preprocessing**:
  - Tokenization using **BERT tokenizer**
  - Stopword removal and stemming using **NLTK**
  - Addressing class imbalance using **SMOTE**
  
- **Feature Engineering**:
  - Text length, character count, word distribution, etc.

- **Models Trained**:
  - **Machine Learning Models**:
    - Random Forest
    - Support Vector Machine (SVM)
  - **Deep Learning Models**:
    - Convolutional Neural Network (CNN)
    - Long Short-Term Memory (LSTM)

- **Performance Evaluation**:
  - **CNN achieved the highest accuracy** among all models.

## 🛠 Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow/Keras
- BERT Tokenizer (from Transformers library)
- NLTK (for text preprocessing)
- Imbalanced-learn (for SMOTE)
- Matplotlib & Seaborn (for visualization)


## 📊 Results
- **CNN model achieved the highest accuracy of 95.86**, outperforming traditional ML models.
- The model generalizes well on unseen text data.

## 🏗️ Future Improvements
- Fine-tuning BERT for end-to-end classification.
- Expanding dataset with more diverse AI-generated text.
- Developing a web-based interface for real-time text classification.



