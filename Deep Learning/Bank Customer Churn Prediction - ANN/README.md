# **Bank Customer Churn Predictor**

This project predicts the likelihood of a customer leaving a bank (churn) based on their personal and account details such as credit score and tenure. The model utilizes an **Artificial Neural Network (ANN)** to identify patterns and insights from the dataset, enabling proactive decision-making to retain customers.

---

## **Project Overview**
Customer churn is a critical metric for financial institutions as retaining customers is often more cost-effective than acquiring new ones. This project leverages an ANN model to predict customer churn based on their personal details, demographics, and account activities.  

The dataset used contains customer details such as age, geography, credit score, and tenure, among others, to assess the probability of churn.

---

## **Dataset Structure**
The dataset is organized as follows:

```
dataset/
├── Churn_Modelling.csv
```

- **Columns Include:**
  - **RowNumber, CustomerId, Surname:** Identifiers for individual customers.
  - **CreditScore, Geography, Gender, Age, Tenure:** Customer demographics and personal details.
  - **Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary:** Account and behavioral metrics.
  - **Exited:** Target variable indicating churn (1 for churn, 0 for retained).

---

## **Data Preprocessing**
To ensure the dataset is ready for training, the following preprocessing steps were applied:

1. **Feature Scaling:**
   - Applied Min-Max scaling to normalize all numerical features to a range between 0 and 1.

2. **Encoding Categorical Variables:**
   - Used **Label Encoder** for binary categorical variables (e.g., Gender).
   - Used **One Hot Encoding** for multi-class categorical variables (e.g., Geography).

3. **Data Splitting:**
   - Split the dataset into training and test sets, with 80% used for training and 20% for testing.

---

## **Model Architecture**
The ANN model consists of the following:

1. **Input Layer:**
   - Inputs: Customer features after preprocessing.

2. **Hidden Layers:**
   - **Hidden Layer 1:** 6 neurons with ReLU activation.
   - **Hidden Layer 2:** 6 neurons with ReLU activation.

3. **Output Layer:**
   - Single neuron with a **Sigmoid activation function** to output the probability of churn.

### **Layer Breakdown:**
| Layer            | Neurons | Activation Function |
|------------------|---------|---------------------|
| Hidden Layer 1   | 6       | ReLU                |
| Hidden Layer 2   | 6       | ReLU                |
| Output Layer     | 1       | Sigmoid             |

---

## **Implementation Details**
- **Programming Language:** Python
- **Libraries Used:**
  - `NumPy`: For numerical computations.
  - `Pandas`: For data manipulation.
  - `TensorFlow`: For building and training the ANN.
- **Training Configuration:**
  - Optimizer: Adam
  - Loss Function: Binary Cross-Entropy
  - Batch Size: 32
  - Epochs: 100

---

## **Results and Performance**
- **Accuracy after 100 epochs:** 86% (on the test set).  
- **Loss trends:** The model effectively minimized loss over epochs to 0.33, demonstrating convergence.  

---

## **How to Use**
### **Prerequisites**
1. Python 3.7+
2. Install the required libraries:
   ```bash
   pip install numpy pandas tensorflow
   ```
   
---

## **Future Improvements**
1. Experiment with additional features like transaction history or behavioral data.
2. Use advanced architectures, such as deeper ANNs or ensemble models, to enhance performance.
3. Add model explainability (e.g., SHAP or LIME) to interpret predictions.




