# **Sentiment Analysis of IMDB Movie Reviews**

This project implements a **Sentiment Analysis** pipeline using machine learning models to classify movie reviews from IMDB as either positive or negative. Two models were trained and evaluated: **Multinomial Naive Bayes (MNB)** and **Random Forest (RF)**, with the Random Forest achieving the best performance.

---

## **Project Overview**

Sentiment analysis is a natural language processing (NLP) technique to determine the sentiment (positive or negative) of text data. This project processes text reviews, converts them into numerical representations, and applies machine learning models for classification.  

**Key Highlights**:  
- Preprocessing pipeline for text cleaning and transformation.  
- Implementation of MNB and RF models.  
- Evaluation using the **AUC-ROC** metric.  
- Ensemble approach for improved accuracy.  

---

## **Programming Language and Libraries**

- **Programming Language**: Python  
- **Libraries Used**:  
  - `bs4`: Removing HTML tags from text.  
  - `pandas`: Data manipulation and analysis.  
  - `re`: Regular expressions for text cleaning.  
  - `nltk`: Natural Language Toolkit for tokenization and stopword removal.  

---

## **Dataset Structure**

The dataset consists of two files:  
1. **Training Data**: `labeledTrainData.csv`  
2. **Testing Data**: `testData.tsv`  

| **Column**   | **Description**      |  
|--------------|----------------------|  
| `review`     | Text of the movie review. |  
| `sentiment`  | Target variable (positive or negative). |  

**Dataset Source**: [IMDB Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  

---

## **Text Preprocessing**

Text preprocessing is critical for NLP tasks. This project uses the following pipeline:  

### **Function Definitions**  

1. **`text_to_words(text)`**:  
   - Removes HTML tags using BeautifulSoup.  
   - Strips non-alphabetic characters using `re.sub`.  
   - Converts text to lowercase for uniformity.  
   - Tokenizes the text into words.  
   - Removes common stopwords using NLTK.  
   - Joins cleaned words back into a string.  

2. **`clean(a)`**:  
   - Applies `text_to_words` on arrays of reviews (e.g., training and testing datasets).  

---

## **Data Preprocessing Pipeline**

1. **Vectorization**:  
   - Used `CountVectorizer` to convert cleaned text into numerical features.  
   - Parameters:  
     - `max_df=0.5`: Ignores words appearing in more than 50% of documents.  
     - `max_features=10000`: Retains the top 10,000 most frequent words.  

2. **Data Splitting**:  
   - Split data into training and validation sets using `train_test_split`.  
   - Features (`Xtrain`, `Xtest`) and target labels (`ytrain`, `ytest`).  

---

## **Model Training and Evaluation**

### **1. Multinomial Naive Bayes (MNB)**  

- **Training**:  
  Trained using the bag-of-words representation of text.  
- **Evaluation**:  
  - `roc_auc_score(ytest, y_val_m)` computes the AUC-ROC metric.  
  - **Validation AUC-ROC**: **0.9238**  

### **2. Random Forest (RF)**  

- **Training**:  
  - Trained with 300 trees (`n_estimators=300`) and the Gini criterion.  
- **Evaluation**:  
  - `roc_auc_score(ytest, y_val_f)` computes the AUC-ROC metric.  
  - **Validation AUC-ROC**: **0.9395**  

### **3. Ensemble Model**  

- Combined predictions from both models to achieve better performance.  
- **Final AUC-ROC**: **0.9337**  

---

## **Results**

| **Model**           | **AUC-ROC Score** |  
|----------------------|-------------------|  
| Multinomial Naive Bayes (MNB) | 0.9238            |  
| Random Forest (RF)   | 0.9395            |  
| Ensemble             | 0.9337            |  

---

## **How to Use**

### **Prerequisites**

1. Python 3.7+  
2. Install required libraries:  
   ```bash
   pip install bs4 pandas nltk
   ```  

---

## **Future Improvements**

1. **Hyperparameter Tuning**:  
   - Experiment with `alpha` values for MNB.  
   - Optimize the number of trees and splitting criteria for RF.  

2. **Additional Models**:  
   - Implement **Logistic Regression** or **SVM** for comparison.  
   - Explore **deep learning models** such as LSTMs or Transformers for better accuracy.  

3. **Advanced Preprocessing**:  
   - Incorporate **lemmatization** or **stemming** to improve feature extraction.  
