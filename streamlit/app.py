import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Sample data for demonstration
data = {
    '# Date': pd.date_range(start='2022-01-01', end='2022-01-10'),
    'Receipt_Count': np.random.randint(500, 1000, size=10)
}

df = pd.DataFrame(data)

# Streamlit app
st.set_page_config(page_title='Receipt Count Predictions', page_icon='📊', layout='wide')

# Title and header
st.title('Receipt Count Predictions for 2022')
st.header('Explore and visualize the receipt count predictions.')

# Display the sample data
st.subheader('Sample Data:')
st.dataframe(df)

# Interactive plot using Altair
st.subheader('Interactive Plot of Receipt Counts over Time:')
chart = alt.Chart(df).mark_line(color='orange').encode(
    x='# Date:T',
    y='Receipt_Count:Q',
    tooltip=['# Date:T', 'Receipt_Count:Q']
).properties(
    width=800,
    height=400
)

st.altair_chart(chart, use_container_width=True)
