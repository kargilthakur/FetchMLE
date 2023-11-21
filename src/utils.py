import pandas as pd
import numpy as np
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from prophet import Prophet
from sklearn.model_selection import ParameterGrid
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def load_data(file_path):
    """
    Load data from a CSV file and convert date to datetime format.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Loaded data.
    """
    df = pd.read_csv(file_path)
    df["# Date"] = pd.to_datetime(df["# Date"])
    return df


def preprocess_date(df):
    """
    Split the date to create additional date features.

    Parameters:
    - df (pd.DataFrame): Input data.

    Returns:
    - pd.DataFrame: Processed data.
    """

    df["Month"] = df["# Date"].dt.month
    df["Year"] = df["# Date"].dt.year
    df["Day"] = df["# Date"].dt.day
    df["DayOfWeek"] = df["# Date"].dt.dayofweek
    df["NumDaysInMonth"] = df["# Date"].dt.daysinmonth

    return df


def add_fred_data(df, indicator, start_date, end_date, frequency="M"):
    """
    Add economic data from FRED to the DataFrame and adjust for the number of days in each month or quarter.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - indicator (str): FRED indicator symbol (e.g., 'GDP', 'UNRATE').
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - end_date (str): End date in 'YYYY-MM-DD' format.
    - frequency (str): Frequency of FRED data ('M' for monthly, 'Q' for quarterly).

    Returns:
    - pd.DataFrame: DataFrame with FRED data added and adjusted for the number of days in each month or quarter.
    """
    fred_data = pdr.get_data_fred(indicator, start=start_date, end=end_date)

    if frequency == "Q":
        quarters_in_year = 4
        dates = pd.date_range(start=start_date, end=end_date)
        fred_df = pd.DataFrame(index=dates, columns=[indicator])

        for i in range(quarters_in_year):
            quarter_mask = dates.quarter == i + 1
            num_days_in_quarter = quarter_mask.sum()
            fred_df.loc[quarter_mask, indicator] = (
                fred_data.iloc[i][indicator] * num_days_in_quarter
            )

    elif frequency == "M":
        months_in_year = 12
        dates = pd.date_range(start=start_date, end=end_date)
        fred_df = pd.DataFrame(index=dates, columns=[indicator])

        for i in range(months_in_year):
            month_mask = dates.month == i + 1
            num_days_in_month = month_mask.sum()
            fred_df.loc[month_mask, indicator] = (
                fred_data.iloc[i][indicator] * num_days_in_month
            )

    merged_df = pd.merge(df, fred_df, left_on="# Date", right_index=True, how="left")
    merged_df[indicator] = merged_df[indicator].ffill()

    return merged_df


def split_data_by_date(df, target_column="# Date", test_start_date="2021-10-01"):
    """
    Split data into training and test sets based on the date.

    Parameters:
    - df (pd.DataFrame): Input data with a date column.
    - target_column (str): Name of the target column.
    - test_start_date (str): Start date for the test set.

    Returns:
    - pd.DataFrame: Training features.
    - pd.DataFrame: Test features.
    - pd.Series: Training target.
    - pd.Series: Test target.
    """
    df["# Date"] = pd.to_datetime(df["# Date"])

    train = df[df["# Date"] < test_start_date]
    test = df[df["# Date"] >= test_start_date]

    X_train, y_train = (
        train.drop([target_column, "Receipt_Count"], axis=1),
        train["Receipt_Count"],
    )
    X_test, y_test = (
        test.drop([target_column, "Receipt_Count"], axis=1),
        test["Receipt_Count"],
    )

    return X_train, X_test, y_train, y_test

def split_data_prophet(df, target_column="# Date", test_start_date="2021-10-01"):
    """
    Split data into training and test sets based on the date.

    Parameters:
    - df (pd.DataFrame): Input data with a date column.
    - target_column (str): Name of the target column.
    - test_start_date (str): Start date for the test set.

    Returns:
    - pd.DataFrame: Training features.
    - pd.DataFrame: Test features.
    - pd.Series: Training target.
    - pd.Series: Test target.
    """
    df["# Date"] = pd.to_datetime(df["# Date"])

    train = df[df["# Date"] < test_start_date]
    test = df[df["# Date"] >= test_start_date]

    return train, test

def evaluate_predictions(actual, predicted):
    """
    Evaluate regression predictions using RMSE, MAE, and R2 score.

    Parameters:
    - actual (array-like): Actual values.
    - predicted (array-like): Predicted values.

    Returns:
    - float: Root Mean Squared Error (RMSE).
    - float: Mean Absolute Error (MAE).
    - float: R2 score.
    """
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared (R2): {r2}")


def plot_actual_vs_predicted(
    actual, predicted, title="Actual vs Predicted", xlabel="Date", ylabel="Value"
):
    """
    Plot actual vs predicted values on a timeline.

    Parameters:
    - actual (array-like): Actual values.
    - predicted (array-like): Predicted values.
    - title (str): Title for the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label="Actual", marker="o")
    plt.plot(predicted, label="Predicted", marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def train_linear_regression_ols(X_train, y_train):
    """
    Train Linear Regression using Ordinary Least Squares (OLS).

    Parameters:
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training target.

    Returns:
    - OLS: Trained OLS model.
    """
    X_train_ols = add_constant(X_train)
    model = OLS(y_train, X_train_ols).fit()
    return model


def get_predictions_2022(model):
    """
    Get predictions for the year 2022 using the specified model.

    Parameters:
    - model: Trained model.

    Returns:
    - pd.Series: Predictions for the year 2022.
    """
    date_range_2022 = pd.date_range(start="2022-01-01", end="2022-12-31", freq="D")
    predictions = []

    for date in date_range_2022:
        date_features = preprocess_date(pd.DataFrame({"# Date": [date]}))
        date_features = date_features.drop("# Date", axis=1).values.flatten()
        prediction = model.predict([date_features])[0]
        predictions.append(prediction)

    return pd.Series(predictions, index=date_range_2022)


def plot_actual_vs_predicted2022(
    actual_2021,
    predicted_2022,
    title="Actual vs Predicted",
    xlabel="Date",
    ylabel="Value",
):
    """
    Plot actual values of 2021 and predicted values of 2022 on a timeline.

    Parameters:
    - actual_2021 (pd.Series): Actual values for the year 2021.
    - predicted_2022 (pd.Series): Predicted values for the year 2022.
    - title (str): Title for the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(actual_2021.index, actual_2021.values, label="Actual 2021", marker="o")
    plt.plot(
        predicted_2022.index, predicted_2022.values, label="Predicted 2022", marker="o"
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def fit_prophet(df):
    """
    Fit a time series model using Prophet.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with columns '# Date' and 'Receipt_Count'.

    Returns:
    - pd.Series: Predictions for the test set.
    """
    df.rename(columns={"# Date": "ds", "Receipt_Count": "y"}, inplace=True)

    model = Prophet(yearly_seasonality=False)
    model.add_seasonality(name="weekly", period=7, fourier_order=3)
    model.add_country_holidays(country_name="US")
    model.fit(df)

    return model


def get_prophet_predictions(model, time_frame, freq ):
    """
    Get monthly aggregated predictions from a Prophet model for a specified time frame.

    Parameters:
    - model: Trained Prophet model.
    - time_frame (int): Number of periods into the future to predict.

    Returns:
    - pd.Series: Monthly aggregated predictions for the specified time frame with the date as the index.
    """
    future = model.make_future_dataframe(freq=freq,periods=time_frame)

    forecast = model.predict(future)
    model.plot_components(forecast)

    predictions_df = forecast.tail(time_frame)[["ds", "yhat"]].rename(
        columns={"ds": "Date", "yhat": "Predicted"}
    )

    predictions_df.set_index("Date", inplace=True)
    return predictions_df

def scale_data(X_train, X_test, y_train, y_test):
    """
    Scale the features and target variable using MinMaxScaler.

    Parameters:
    - X_train (pd.DataFrame): Training features.
    - X_test (pd.DataFrame): Testing features.
    - y_train (pd.Series): Training target.

    Returns:
    - tuple: Scaled training features, scaled testing features, scaled training target, scaler for the target.
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.fit_transform(y_test.reshape(-1, 1)).flatten()

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y

def reshape_data(X_train_scaled, X_test_scaled):
    """
    Reshape the input data for LSTM.

    Parameters:
    - X_train_scaled (np.ndarray): Scaled training features.
    - X_test_scaled (np.ndarray): Scaled testing features.

    Returns:
    - tuple: Reshaped training features, reshaped testing features.
    """
    X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
    return X_train_reshaped, X_test_reshaped

def create_lstm_model(input_shape=(None, 1)):
    """
    Create a basic LSTM model.

    Parameters:
    - input_shape (tuple): Input shape for the model.

    Returns:
    - Sequential: Basic LSTM model.
    """
    model = Sequential()
    model.add(LSTM(100, input_shape=input_shape))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=0.001)  
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def grid_search_lstm(X_train_reshaped, y_train_scaled, X_val_reshaped, y_val_scaled, param_grid):
    """
    Perform grid search for LSTM hyperparameters.

    Parameters:
    - X_train_reshaped (numpy.ndarray): Reshaped training features.
    - y_train_scaled (numpy.ndarray): Scaled training target.
    - X_val_reshaped (numpy.ndarray): Reshaped validation features.
    - y_val_scaled (numpy.ndarray): Scaled validation target.
    - param_grid (dict): Hyperparameter grid.

    Returns:
    - dict: Best hyperparameters and corresponding model.
    """
    best_rmse = float('inf')
    best_params = {}

    for params in ParameterGrid(param_grid):
        model = create_lstm_model(units=params['units'], learning_rate=params['learning_rate'], input_shape=(X_train_reshaped.shape[1], 1))
        model.fit(X_train_reshaped, y_train_scaled, epochs=100, batch_size=32, validation_data=(X_val_reshaped, y_val_scaled), verbose=0)

        lstm_predictions_scaled = model.predict(X_val_reshaped)
        lstm_predictions = lstm_predictions_scaled.flatten()

        lstm_rmse = np.sqrt(mean_squared_error(y_val_scaled, lstm_predictions))

        if lstm_rmse < best_rmse:
            best_rmse = lstm_rmse
            best_params = params
            best_model = model

    return {'best_params': best_params, 'best_model': best_model}

def train_lstm_model(X_train_reshaped, y_train_scaled, X_test_reshaped, y_test_scaled, best_params, epochs=100, batch_size=32):
    """
    Train LSTM model using the best hyperparameters.

    Parameters:
    - X_train_reshaped (numpy.ndarray): Reshaped training features.
    - y_train_scaled (numpy.ndarray): Scaled training target.
    - X_test_reshaped (numpy.ndarray): Reshaped test features.
    - y_test_scaled (numpy.ndarray): Scaled test target.
    - best_params (dict): Best hyperparameters.
    - epochs (int): Number of epochs for training.
    - batch_size (int): Batch size for training.

    Returns:
    - Sequential: Trained LSTM model.
    """
    model = create_lstm_model(learning_rate=best_params['learning_rate'], units=best_params['units'])
    model.fit(X_train_reshaped, y_train_scaled, epochs=epochs, batch_size=batch_size, validation_data=(X_test_reshaped, y_test_scaled), verbose=2)

    return model

def get_lstm_predictions(model, X_future_reshaped, scaler_y):
    """
    Get predictions from the trained LSTM model.

    Parameters:
    - model: Trained LSTM model.
    - X_future_reshaped (numpy.ndarray): Reshaped features for future predictions.
    - scaler_y: Scaler for the target variable.

    Returns:
    - numpy.ndarray: Predictions.
    """
    lstm_predictions_scaled = model.predict(X_future_reshaped)
    lstm_predictions = scaler_y.inverse_transform(lstm_predictions_scaled).flatten()

    return lstm_predictions

def get_lstm_predictions_2022(final_model, scaler_X, scaler_y):
    """
    Get predictions for the year 2022 using the trained LSTM model.

    Parameters:
    - final_model: Trained LSTM model.
    - scaler_X: Scaler used for input features.
    - scaler_y: Scaler used for target variable.

    Returns:
    - pd.DataFrame: Monthly predictions for 2022 with the date and prediction columns.
    """
    future_dates_2022 = pd.date_range(start='2022-01-01', end='2022-12-31', freq='M')
    
    future_data_2022 = pd.DataFrame({
        'Month': future_dates_2022.month,
        'Day': future_dates_2022.day,
        'DayOfWeek': future_dates_2022.dayofweek
    })
    
    future_data_2022_scaled = scaler_X.transform(future_data_2022)
    
    future_data_2022_reshaped = future_data_2022_scaled.reshape(
        (future_data_2022_scaled.shape[0], future_data_2022_scaled.shape[1], 1)
    )
    
    lstm_predictions_2022_scaled = final_model.predict(future_data_2022_reshaped)
    
    lstm_predictions_2022 = scaler_y.inverse_transform(lstm_predictions_2022_scaled).flatten()
    
    monthly_predictions_2022 = pd.DataFrame({
        'Date': future_dates_2022,
        'Prediction': lstm_predictions_2022
    })

    return monthly_predictions_2022

