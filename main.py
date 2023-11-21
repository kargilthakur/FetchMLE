from src.utils import *
from configparser import ConfigParser
import pickle
from datetime import datetime


def main():
    config = ConfigParser()
    config.read("config/config.ini")
    data = config["models"]
    model_choice = "Prophet"
    df = load_data("data/data_daily.csv")

    if model_choice == "Linear":
        df = preprocess_date(df)
        X_train, X_test, y_train, y_test = split_data_by_date(df)
        model = train_linear_regression_ols(X_train, y_train)
        y_pred = model.predict(X_test)
        evaluate_predictions(y_test, y_pred)
        # plot_actual_vs_predicted(y_test,y_pred)
        predictions = get_predictions_2022(model)
        actual = load_data("data/data_daily.csv")
        actual = actual.set_index("# Date").squeeze()
        plot_actual_vs_predicted2022(actual, predictions)
        predictions.to_csv(
            f'data/predictions_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv',
            header=True,
        )
        with open("models/LinearRegression.pkl", "wb") as file:
            pickle.dump(model, file)

    if model_choice == "Prophet":
        df = add_fred_data(df, "PCEPI", "2021-01-01", "2021-12-31", "M")
        model = fit_prophet(df)
        predictions = get_predictions_2022(model, 365)
        actual = load_data("data/data_daily.csv")
        actual = actual.set_index("# Date").squeeze()
        plot_actual_vs_predicted2022(actual, predictions)
        predictions.to_csv(
            f'data/predictions_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv',
            header=True,
        )
        with open("models/Prophet.pkl", "wb") as file:
            pickle.dump(model, file)


if __name__ == "__main__":
    main()
