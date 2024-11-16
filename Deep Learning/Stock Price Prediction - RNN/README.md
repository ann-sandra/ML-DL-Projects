# **Stock Price Predictor**

This project implements a model to predict the next day’s **Google stock price** based on the past 60 days' stock prices. The model utilizes a **Long Short-Term Memory (LSTM) Recurrent Neural Network (RNN)**, which is specifically designed to capture temporal dependencies in sequential data.

---

## **Project Overview**
Stock price prediction is a challenging problem due to the dynamic and complex nature of financial markets. This project leverages **LSTM RNNs**, which are well-suited for time-series forecasting tasks, to predict stock prices by learning patterns and trends in historical data.

The model takes the **opening stock prices** from the last 60 days as input and predicts the **next day's stock price**.

---

## **Dataset Structure**
The dataset consists of two CSV files:
```
dataset/
├── Google_Stock_Price_Training.csv
├── Google_Stock_Price_Test.csv
```

- **Training Set:** Used to train the model, containing historical stock prices.
- **Test Set:** Used to evaluate the model’s performance on unseen data.

---

## **Model Architecture**
The **RNN** used in this project has the following architecture:

1. **Input Sequence:**  
   - 60 time steps (past 60 days’ opening prices).
   - Feature scaling applied to normalize data to a range of 0 to 1.

2. **LSTM Layers:**
   - Four LSTM layers, each with **50 neurons** and a **20% dropout rate** for regularization.
   - The dropout helps prevent overfitting by randomly disabling neurons during training.

3. **Output Layer:**
   - A single neuron in the output layer serves as the regressor to predict the next day's stock price.

---

## **Implementation Details**
- **Programming Language:** Python
- **Libraries Used:**
  - `NumPy`: For numerical computations.
  - `Pandas`: For data manipulation and preprocessing.
  - `Matplotlib`: For data visualization.
  - `Keras`: For building and training the LSTM RNN.
- **Training Configuration:**
  - Batch Size: 32
  - Epochs: 100
  - Loss Function: Mean Squared Error (MSE)
  - Optimizer: Adam
- **Data Preprocessing:**
  - Normalized data between 0 and 1 using Min-Max Scaling.
  - Created sequences of 60 time steps for training.

---

## **Results and Performance**
- **Accuracy after 100 epochs:** 83% (on the test set).  
- **Loss trends:** The model showed consistent improvement in minimizing loss across epochs.  

---

## **Visualisation fo Real and Predicted Google Stock Price**
![Real vs Predicted Stock Price](Real%20vs%20Predicted%20Stock%20Price.png)

---

## **How to Use**
### **Prerequisites**
1. Python 3.7+
2. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib keras
   ```

---

## **Future Improvements**
1. Experiment with additional features like closing prices, trading volume, or moving averages.
2. Use different architectures (e.g., GRU or Bidirectional LSTMs).
4. Extend the model to predict stock prices over a longer horizon (e.g., next 7 days).

---

