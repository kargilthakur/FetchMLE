import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import plotly.express as px

# Streamlit app
st.set_page_config(page_title="Receipt Count Predictions", page_icon="📊", layout="wide")

# Title and header
st.title("Receipt Count Predictions for 2022")
st.header("Explore and visualize the receipt count predictions.")

# Model selection
model = st.selectbox(
    'Which model predictions do you want to see?',
    ('LSTM', 'Linear Regression', 'Prophet')  # LSTM is now the first option
)

# Load the corresponding CSV file
if model == 'Linear Regression':
    csv_path = '/Users/kargilthakur/FetchMLE/data/lr_predictions_20231121154304.csv'
elif model == 'LSTM':
    csv_path = '/Users/kargilthakur/FetchMLE/data/lstm_predictions_20231121160726.csv'
else:
    csv_path = '/Users/kargilthakur/FetchMLE/data/lstm_predictions_20231121160726.csv'

df = pd.read_csv(csv_path, parse_dates=True, index_col=0)

# Rename the column
df.rename(columns={'0': 'Predicted'}, inplace=True)

# Group by month and sum
df_monthly = df.resample('M').sum()

# Generate button
if st.button('Generate'):
    # Display the sample data
    st.subheader("Receipt counts:")
    st.dataframe(df_monthly)

    # Interactive plot using Plotly
    st.subheader("Interactive Plot of Receipt Counts over Time:")
    fig = px.line(df_monthly, x=df_monthly.index, y="Predicted")
    st.plotly_chart(fig)

    # Download data as CSV
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings
    href = f'<a href="data:file/csv;base64,{b64}" download="daily_data.csv">Download daily data CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)
