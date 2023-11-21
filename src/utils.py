import pandas as pd
import numpy as np
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX

def load_data(file_path):
    """
    Load data from a CSV file and convert date to datetime format.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Loaded data.
    """
    df = pd.read_csv(file_path)
    df['# Date'] = pd.to_datetime(df['# Date'])
    return df

def preprocess_date(df):
    """
    Split the date to create additional date features.

    Parameters:
    - df (pd.DataFrame): Input data.

    Returns:
    - pd.DataFrame: Processed data.
    """
    # Extract date features
    df['Month'] = df['# Date'].dt.month
    df['Year'] = df['# Date'].dt.year
    df['Day'] = df['# Date'].dt.day
    df['DayOfWeek'] = df['# Date'].dt.dayofweek
    df['NumDaysInMonth'] = df['# Date'].dt.daysinmonth

    return df


def add_fred_data(df, indicator, start_date, end_date, frequency='M'):
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

    if frequency == 'Q':
        quarters_in_year = 4
        dates = pd.date_range(start=start_date, end=end_date)
        fred_df = pd.DataFrame(index=dates, columns=[indicator])
        
        for i in range(quarters_in_year):
            quarter_mask = (dates.quarter == i + 1)
            num_days_in_quarter = quarter_mask.sum()
            fred_df.loc[quarter_mask, indicator] = fred_data.iloc[i][indicator] * num_days_in_quarter

    elif frequency == 'M':
        months_in_year = 12
        dates = pd.date_range(start=start_date, end=end_date)
        fred_df = pd.DataFrame(index=dates, columns=[indicator])

        for i in range(months_in_year):
            month_mask = (dates.month == i + 1)
            num_days_in_month = month_mask.sum()
            fred_df.loc[month_mask, indicator] = fred_data.iloc[i][indicator] * num_days_in_month

    merged_df = pd.merge(df, fred_df, left_on='# Date', right_index=True, how='left')
    merged_df[indicator] = merged_df[indicator].ffill()

    return merged_df

def split_data_by_date(df, target_column='# Date', test_start_date='2021-10-01'):
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
    df['# Date'] = pd.to_datetime(df['# Date'])

    # Split data based on the date
    train = df[df['# Date'] < test_start_date]
    test = df[df['# Date'] >= test_start_date]

    # Extract features and target
    X_train, y_train = train.drop([target_column, 'Receipt_Count'], axis=1), train['Receipt_Count']
    X_test, y_test = test.drop([target_column, 'Receipt_Count'], axis=1), test['Receipt_Count']

    return X_train, X_test, y_train, y_test

def scale_data(df, column_name):
    """
    Scale a specific column in the data using Min-Max scaling.

    Parameters:
    - df (pd.DataFrame): Input data.
    - column_name (str): Name of the column to be scaled.

    Returns:
    - pd.DataFrame: Scaled data.
    - MinMaxScaler: Scaler for the specified column.
    """
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[column_name] = scaler.fit_transform(df[column_name].values.reshape(-1, 1)).flatten()
    return df_scaled, scaler

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

    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'R-squared (R2): {r2}')

def plot_actual_vs_predicted(actual, predicted, title='Actual vs Predicted', xlabel='Date', ylabel='Value'):
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
    plt.plot(actual, label='Actual', marker='o')
    plt.plot(predicted, label='Predicted', marker='o')
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
    # Generate a date range for the year 2022
    date_range_2022 = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')

    predictions = []

    for date in date_range_2022:
        # Extract date features using preprocess_date function
        date_features = preprocess_date(pd.DataFrame({'# Date': [date]}))
        
        # Remove the '# Date' column and convert to a 1D array
        date_features = date_features.drop('# Date', axis=1).values.flatten()

        # Predict using the model
        prediction = model.predict([date_features])[0]
        predictions.append(prediction)

    return pd.Series(predictions, index=date_range_2022)

def plot_actual_vs_predicted2022(actual_2021, predicted_2022, title='Actual vs Predicted', xlabel='Date', ylabel='Value'):
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
    plt.plot(actual_2021.index, actual_2021.values, label='Actual 2021', marker='o')
    plt.plot(predicted_2022.index, predicted_2022.values, label='Predicted 2022', marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def train_random_forest_with_hyperparameter_tuning(X_train, y_train, param_grid, cv=3):
    """
    Train Random Forest with hyperparameter tuning using Grid Search.

    Parameters:
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training target.
    - param_grid (dict): Dictionary of hyperparameter values to search.
    - cv (int): Number of cross-validation folds.

    Returns:
    - RandomForestRegressor: Trained Random Forest model with best hyperparameters.
    """
    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=cv)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print(f'Best Hyperparameters: {best_params}')

    final_model = RandomForestRegressor(random_state=42, **best_params)
    final_model.fit(X_train, y_train)

    return final_model


def train_random_forest(X_train, y_train, use_hyperparameter_tuning=False):
    """
    Train Random Forest.

    Parameters:
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training target.
    - use_hyperparameter_tuning (bool): Whether to perform hyperparameter tuning.

    Returns:
    - RandomForestRegressor: Trained Random Forest model.
    """
    if use_hyperparameter_tuning:
        param_grid = {
            'n_estimators': [20, 30, 40],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        model = train_random_forest_with_hyperparameter_tuning(X_train, y_train, param_grid)
    else:
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

    return model

def train_sarima(X_train, y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    """
    Train a time series model using SARIMA.

    Parameters:
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training target.
    - order (tuple): Order of the autoregressive, differencing, and moving average components.
    - seasonal_order (tuple): Order of the seasonal autoregressive, differencing, and moving average components.

    Returns:
    - SARIMAX: Trained SARIMA model.
    """
    model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    result = model.fit(disp=False)
    return result

def fit_prophet(df):
    """
    Fit a time series model using Prophet.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with columns '# Date' and 'Receipt_Count'.

    Returns:
    - pd.Series: Predictions for the test set.
    """
    # Rename columns for Prophet
    df.rename(columns={'# Date': 'ds', 'Receipt_Count': 'y'}, inplace=True)

    df = preprocess_date(df)
    train, test = split_data(df)

    # Initialize the Prophet model with US holidays
    model = Prophet(yearly_seasonality=False)  # Disable yearly seasonality
    model.add_regressor('Month', standardize=False)
    model.add_seasonality(name='weekly', period=7, fourier_order=3)  # Add weekly seasonality
    model.add_seasonality(name='monthly', period=30.44, fourier_order=5)  # Add monthly seasonality

    # Fit the model
    model.fit(train)

    return model
    # Create a dataframe with future dates for prediction

def get_prophet_predictions(model, time_frame):
    """
    Get predictions from a Prophet model for a specified time frame.

    Parameters:
    - model: Trained Prophet model.
    - time_frame (int): Number of periods into the future to predict.

    Returns:
    - pd.Series: Predictions for the specified time frame.
    """
    future = model.make_future_dataframe(periods=time_frame)
    future['Month'] = future['ds'].dt.month

    forecast = model.predict(future)

    forecast['yhat'] = forecast['yhat']

    predictions = forecast.tail(time_frame)['yhat'].values

    return pd.Series(predictions)