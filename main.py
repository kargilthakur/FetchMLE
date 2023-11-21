from src.utils import *
from configparser import ConfigParser
import pickle
from datetime import datetime


def main():
    config = ConfigParser()
    config.read("config/config.ini")
    data = config["models"]
    model_choice = "LSTM"
    df = load_data("data/data_daily.csv")

    if model_choice == "Linear":
        df = preprocess_date(df)
        X_train, X_test, y_train, y_test = split_data_by_date(df)
        model = train_linear_regression_ols(X_train, y_train)
        y_pred = model.predict(X_test)
        evaluate_predictions(y_test, y_pred)

        with open("models/LinearRegression.pkl", "wb") as file:
            pickle.dump(model, file)

        predictions = get_predictions_2022(model)
        actual = load_data("data/data_daily.csv")
        actual = actual.set_index("# Date").squeeze()
        plot_actual_vs_predicted2022(actual, predictions)

        predictions.to_csv(
            f'data/predictions_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv',
            header=True,
        )

    elif model_choice == "Prophet":
        train, test = split_data_prophet(df)
        print(len(test))
        model = fit_prophet(train)
        predictions = get_prophet_predictions(model, 92, "D")
        evaluate_predictions(test["Receipt_Count"], predictions)

        with open("models/Prophet.pkl", "wb") as file:
            pickle.dump(model, file)

        predictions2022 = get_prophet_predictions(model, 15, "M")
        actual = load_data("data/data_daily.csv")
        actual = actual.set_index("# Date").squeeze()
        actual = actual.resample("M").last()
        plot_actual_vs_predicted2022(actual, predictions2022[3:])

        predictions.to_csv(
            f'data/predictions_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv',
            header=True,
        )

    elif model_choice == "LSTM":
        df = preprocess_date(df)
        X_train, X_test, y_train, y_test = split_data_by_date(df)
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y = scale_data(X_train, X_test, y_train, y_test)
        X_train_reshaped, X_test_reshaped = reshape_data(X_train_scaled, X_test_scaled)

        param_grid = {
            'learning_rate': [0.001, 0.01, 0.1],
            'units': [50, 100, 150],
            'dropout_rate': [0.2, 0.3, 0.4]
        }

        grid_search_result = grid_search_lstm(X_train_reshaped, y_train_scaled, X_test_reshaped, y_test_scaled, param_grid)
        best_params = grid_search_result['best_params']
        final_model = train_lstm_model(X_train_reshaped, y_train_scaled, X_test_reshaped, y_test_scaled, best_params, epochs=100, batch_size=32)
        monthly_predictions_2022 = get_lstm_predictions_2022(final_model, scaler_X, scaler_y)
        actual = load_data("data/data_daily.csv")
        actual = actual.set_index("# Date").squeeze()
        plot_actual_vs_predicted2022(actual, monthly_predictions_2022)


if __name__ == "__main__":
    main()
