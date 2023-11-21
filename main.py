from src.utils import *
import pandas as pd
from configparser import ConfigParser

def main():
    config = ConfigParser()
    config.read('config/config.ini')
    data = config['models']
    df = load_data('data/data_daily.csv')
    df = preprocess_date(df)
    df = add_fred_data(df,'PCEPI','2021-01-01','2021-12-31','M')
    print(df)
if __name__ == "__main__":
    main()