import pandas as pd
import numpy as np
import unittest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.regression.linear_model import OLS
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

from utils import (
    load_data,
    preprocess_date,
    split_data,
    scale_data,
    evaluate_predictions,
    train_linear_regression_ols,
    train_random_forest,
    train_sarima,
    fit_prophet,
    get_prophet_predictions,
)

class TestUtils(unittest.TestCase):
    def setUp(self):
        # Load example data for testing
        file_path = 'path_to_your_data.csv'
        self.df = load_data(file_path)
        self.df = preprocess_date(self.df)

    def test_split_data(self):
        """Test the split_data function."""
        train, test = split_data(self.df)
        self.assertEqual(len(train) + len(test), len(self.df))
        self.assertIsInstance(train, pd.DataFrame)
        self.assertIsInstance(test, pd.DataFrame)

    def test_scale_data(self):
        """Test the scale_data function."""
        column_name = 'Receipt_Count'
        scaled_df, scaler = scale_data(self.df, column_name)
        self.assertIn(column_name, scaled_df.columns)
        self.assertIsInstance(scaler, MinMaxScaler)

    def test_evaluate_predictions(self):
        """Test the evaluate_predictions function."""
        actual = np.array([1, 2, 3, 4, 5])
        predicted = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
        rmse, mae, r2 = evaluate_predictions(actual, predicted)
        self.assertIsInstance(rmse, float)
        self.assertIsInstance(mae, float)
        self.assertIsInstance(r2, float)

    def test_train_linear_regression_ols(self):
        """Test the train_linear_regression_ols function."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.df[['Month', 'Day', 'DayOfWeek']], self.df['Receipt_Count'], test_size=0.2, random_state=42
        )
        model = train_linear_regression_ols(X_train, y_train)
        self.assertIsInstance(model, OLS)

    def test_train_random_forest(self):
        """Test the train_random_forest function."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.df[['Month', 'Day', 'DayOfWeek']], self.df['Receipt_Count'], test_size=0.2, random_state=42
        )
        model = train_random_forest(X_train, y_train, use_hyperparameter_tuning=False)
        self.assertIsInstance(model, RandomForestRegressor)

    def test_train_sarima(self):
        """Test the train_sarima function."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.df[['Month', 'Day', 'DayOfWeek']], self.df['Receipt_Count'], test_size=0.2, random_state=42
        )
        model = train_sarima(X_train, y_train)
        self.assertIsInstance(model, SARIMAX)

    def test_fit_prophet(self):
        """Test the fit_prophet function."""
        model = fit_prophet(self.df)
        self.assertIsInstance(model, Prophet)

    def test_get_prophet_predictions(self):
        """Test the get_prophet_predictions function."""
        model = fit_prophet(self.df)
        time_frame = 10
        predictions = get_prophet_predictions(model, time_frame)
        self.assertIsInstance(predictions, pd.Series)

if __name__ == '__main__':
    unittest.main()
