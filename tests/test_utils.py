import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

from src.utils import (
    load_data,
    preprocess_date,
    scale_data,
    split_data_by_date,
    reshape_data,
)


class TestUtils(unittest.TestCase):
    def setUp(self):
        # Load example data for testing
        file_path = "data/data_daily.csv"
        self.df = load_data(file_path)

    def test_load_data(self):
        # Test the load_data function
        file_path = "data/data_daily.csv"
        df = load_data(file_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue("# Date" in df.columns)

    def test_preprocess_date(self):
        # Test the preprocess_date function
        df = pd.DataFrame(
            {"# Date": pd.date_range(start="2022-01-01", periods=5, freq="D")}
        )
        processed_df = preprocess_date(df)
        self.assertIsInstance(processed_df, pd.DataFrame)
        self.assertTrue("Month" in processed_df.columns)
        self.assertTrue("Year" in processed_df.columns)
        self.assertTrue("Day" in processed_df.columns)

    def test_split_data_by_date(self):
        # Test the split_data_by_date function
        df = pd.DataFrame(
            {
                "# Date": pd.date_range(start="2022-01-01", periods=10, freq="D"),
                "Receipt_Count": np.arange(10),
            }
        )
        X_train, X_test, y_train, y_test = split_data_by_date(
            df, target_column="# Date", test_start_date="2022-01-06"
        )
        self.assertIsInstance(X_train, pd.DataFrame)
        self.assertIsInstance(X_test, pd.DataFrame)
        self.assertIsInstance(y_train, pd.Series)
        self.assertIsInstance(y_test, pd.Series)

    def test_scale_data(self):
        # Test the scale_data function
        X_train = pd.DataFrame({"Feature1": np.arange(5), "Feature2": np.arange(5, 10)})
        X_test = pd.DataFrame(
            {"Feature1": np.arange(10, 15), "Feature2": np.arange(15, 20)}
        )
        y_train = pd.Series(np.arange(5))
        y_test = pd.Series(np.arange(5, 10))
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, _, _ = scale_data(
            X_train, X_test, y_train, y_test
        )
        self.assertIsInstance(X_train_scaled, np.ndarray)
        self.assertIsInstance(X_test_scaled, np.ndarray)
        self.assertIsInstance(y_train_scaled, np.ndarray)
        self.assertIsInstance(y_test_scaled, np.ndarray)

    def test_reshape_data(self):
        # Test the reshape_data function
        X_train_scaled = np.random.rand(5, 2)
        X_test_scaled = np.random.rand(2, 2)
        X_train_reshaped, X_test_reshaped = reshape_data(X_train_scaled, X_test_scaled)
        self.assertIsInstance(X_train_reshaped, np.ndarray)
        self.assertIsInstance(X_test_reshaped, np.ndarray)
        self.assertEqual(X_train_reshaped.shape[2], 1)
        self.assertEqual(X_test_reshaped.shape[2], 1)


if __name__ == "__main__":
    unittest.main()
