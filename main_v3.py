# Required libraries
# pip install streamlit prophet yfinance plotly scikit-learn pandas statsmodels
import streamlit as st
from datetime import date
import numpy as np
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Define the start date for fetching data and today's date for the end date
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# App title displayed on the Streamlit app
st.title('Adaptive Stock Predictor Web App')

# Dropdown to select stock
stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

# Slider to select the number of years for prediction
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365  # Convert years to days

# Dropdown to select the forecasting model
model_choice = st.selectbox('Select the model for prediction', 
                            ('Prophet', 'Linear Regression', 'Double Exponential Smoothing', 'Triple Exponential Smoothing'))

# Function to load data from Yahoo Finance
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Display state of data loading
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

# Display raw data
st.subheader('Raw data')
st.write(data.tail())

# Function to plot raw stock data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Forecasting with Prophet model
def run_prophet(data):
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)
    st.subheader('Forecast data with Prophet')
    st.write(forecast.tail())
    st.write(f'Forecast plot for {n_years} years using Prophet')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)
    st.write("Forecast components with Prophet")
    fig2 = m.plot_components(forecast)
    st.write(fig2)

# Forecasting with Linear Regression model
def run_linear_regression(data):
    data['Days'] = (data['Date'] - data['Date'].min()).dt.days
    X = data[['Days']]  # Features
    y = data['Close']  # Target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    future_days = np.arange(X['Days'].max() + 1, X['Days'].max() + period + 1).reshape(-1, 1)
    future_forecast = model.predict(future_days)
    st.subheader('Linear Regression forecast data')
    st.write(future_forecast[-5:])
    st.write(f'Forecast plot for {n_years} years using Linear Regression')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=y, name="Actual Close"))
    future_dates = pd.date_range(start=data['Date'].max() + pd.Timedelta(days=1), periods=period, freq='D')
    fig.add_trace(go.Scatter(x=future_dates, y=future_forecast, name="Predicted Close"))
    fig.layout.update(title_text='Forecast with Linear Regression', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# Forecasting with Double Exponential Smoothing model
def run_double_exponential_smoothing(data):
    df_train = data.set_index('Date')['Close']
    model = ExponentialSmoothing(df_train, trend='add', seasonal=None, seasonal_periods=None)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=365 * n_years)
    st.subheader('Double Exponential Smoothing forecast data')
    st.write(forecast.tail())
    st.write('Double Exponential Smoothing forecast plot')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_train.index, y=df_train, name='Actual Close'))
    forecast_index = pd.date_range(start=df_train.index.max() + pd.Timedelta(days=1), periods=365 * n_years, freq='D')
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast, name='Forecast'))
    fig.layout.update(title_text='Double Exponential Smoothing Forecast', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# Forecasting with Triple Exponential Smoothing model
def run_triple_exponential_smoothing(data):
    df_train = data.set_index('Date')['Close']
    model = ExponentialSmoothing(df_train, trend='add', seasonal='add', seasonal_periods=365)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=365 * n_years)
    st.subheader('Triple Exponential Smoothing forecast data')
    st.write(forecast.tail())
    st.write('Triple Exponential Smoothing forecast plot')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_train.index, y=df_train, name='Actual Close'))
    forecast_index = pd.date_range(start=df_train.index.max() + pd.Timedelta(days=1), periods=365 * n_years, freq='D')
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast, name='Forecast'))
    fig.layout.update(title_text='Triple Exponential Smoothing Forecast', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# Execution block to run the chosen model
if model_choice == 'Prophet':
    run_prophet(data)
elif model_choice == 'Linear Regression':
    run_linear_regression(data)
elif model_choice == 'Double Exponential Smoothing':
    run_double_exponential_smoothing(data)
else:
    run_triple_exponential_smoothing(data)
