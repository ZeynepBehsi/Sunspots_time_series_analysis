# Sunspots Time Series Analysis

## Project Overview
This project aims to predict monthly sunspot activity using historical data from 1749 to the present, sourced from the SIDC (Solar Influences Data Analysis Center) database. Three machine learning models were implemented: a Deep Neural Network (DNN), a combination of LSTM and CNN, and a hybrid model using DNN, LSTM, and CNN together. Each model was tuned for optimal performance, with the hybrid model achieving the best results.

## Features
- **Time Series Forecasting**: Predicts future sunspot activity based on historical data.
- **Multiple Models**: Compares three models—DNN, LSTM+CNN, and a hybrid DNN+LSTM+CNN model.
- **Data Normalization**: All models use A Lambda layer to scale the input data.
- **Learning Rate Optimization**: Callbacks were used to monitor and optimize the learning rate during training.

## Technologies Used
- Python
- TensorFlow
- Keras
- Deep Neural Networks (DNN)
- Long Short-Term Memory (LSTM)
- Convolutional Neural Networks (CNN)

## Models
1. **DNN Model**:
   - Mean Absolute Error (MAE): 14.038267
   
2. **LSTM + CNN Model**:
   - Mean Absolute Error (MAE): 14.523124
   
3. **Hybrid Model (DNN + LSTM + CNN)**:
   - Architecture:
     - 1 hidden CNN layer
     - 2 hidden LSTM layers
     - 2 hidden Dense layers
     - Lambda layer for data normalization
     - Dense output layer with 1 unit
   - Mean Absolute Error (MAE): 13.753272

## Installation
1. Clone the repository:
   ```bash
   git cl one https://github.com/yourusername/Sunspots_time_series_analysis.git


## Results
- DNN Model MAE: 14.038267
- LSTM + CNN Model MAE: 14.523124
- Hybrid Model (DNN + LSTM + CNN) MAE: 13.753272
  
  - Learning rate optimization and parameter tuning improved model performance significantly, with the hybrid model yielding the best results.

## Conclusion
This project demonstrates the effectiveness of using deep learning models for time series analysis. The hybrid model, combining DNN, LSTM, and CNN, showed the best performance in predicting future sunspot activity. Learning rate optimization and other tuning techniques were key in achieving the best results.
