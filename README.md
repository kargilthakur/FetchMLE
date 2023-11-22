# FetchMLE

## Problem Statement

At Fetch, we closely monitor the number of scanned receipts in our app on a daily basis, as it serves as a crucial Key Performance Indicator (KPI). In certain business scenarios, it becomes essential to predict the potential number of scanned receipts for a given future month.

The provided dataset contains the observed number of scanned receipts each day throughout the year 2021. The goal is to develop an algorithm capable of predicting the approximate number of scanned receipts for each month in 2022.

## Inference Deployment

The inference process is deployed on Hugging Face, which provides a convenient platform for interaction. You can access the inference platform [here](https://huggingface.co/spaces/kargil8320/MLE). Note that due to resource limitations on the free Hugging Face platform, the deployed app pauses every 48 hours. If you encounter any issues, consider restarting it.

For a more robust inference experience, you can pull the Docker image from the following repository:

```bash
docker pull tkargil0/receiptpredictor:latest
```

This Docker image includes the complete pipeline, including model training, prediction generation, and a Streamlit app for visualization.

## Modelling Approach

Given the limited features in the dataset, additional features were engineered based on the date, including month, day, days in the month, etc. The dataset for the year 2021 was divided into training and testing sets using a 75-25 split.

Three distinct modeling approaches were employed to address this prediction task:

### 1. Linear Problem - Linear Regression using OLS

- **Data Preparation:** The dataset was preprocessed by creating additional date-related features.
- **Model Training:** Linear Regression using Ordinary Least Squares (OLS) was employed for training.
- **Evaluation:** The model was evaluated using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).
- **Prediction:** The model was used to predict the number of scanned receipts for each month in 2022.

### 2. Time Series Problem - Facebook Prophet Model

- **Data Preparation:** The dataset was split into training and test sets for time series modeling.
- **Model Training:** A time series model using Facebook Prophet was fitted to the training data.
- **Evaluation:** The model's performance was evaluated using RMSE and MAE.
- **Prediction:** Predictions were generated for each day in 2022.

### 3. Non-Linear Problem - LSTM Model

- **Data Preparation:** Date features were engineered, and the dataset was split into training and test sets.
- **Model Training:** A Long Short-Term Memory (LSTM) neural network was created and trained using a grid search for hyperparameter tuning.
- **Evaluation:** The LSTM model's performance was evaluated using RMSE.
- **Prediction:** Monthly predictions for 2022 were obtained using the trained LSTM model.

Each method offers a unique perspective on predicting scanned receipts, catering to different aspects of the underlying patterns in the data. The results and predictions can be visualized through the provided Hugging Face inference platform or by running the Docker image locally.
