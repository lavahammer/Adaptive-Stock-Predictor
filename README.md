# Adaptive-Stock-Predictor
Adaptive Stock Predictor: Harnessing Prophet, Linear Regression, and Exponential Smoothing Models in a Streamlit Application

The "Adaptive Stock Predictor" application is designed to provide accurate stock price forecasts using four different models: Prophet, Linear Regression, Double Exponential Smoothing, and Triple Exponential Smoothing. The application is built using Streamlit, which offers a user-friendly interface for interacting with the models and visualizing the results.
Key Components of the Solution
1.	User Interface (UI) with Streamlit:
•	Stock Selection: Users can select from a list of predefined stocks (e.g., GOOG, AAPL, MSFT, GME) using a dropdown menu.
•	Prediction Period: Users can specify the number of years for the prediction using a slider.
•	Model Selection: Users can choose the forecasting model from a dropdown menu.
•	Visualization: The application provides interactive visualizations of historical and predicted stock prices using Plotly.
2.	Data Fetching and Preprocessing:
•	Data Source: Historical stock data is fetched from Yahoo Finance using the yfinance library.
•	Data Cleaning: The data is cleaned and pre-processed to handle missing values and outliers, ensuring it is suitable for modelling.
3.	Forecasting Models:
•	Prophet Model: Used for capturing seasonality and trend components in the time series data.
•	Linear Regression: A simple regression model that predicts future stock prices based on the linear trend observed in historical data.
•	Double Exponential Smoothing: Captures trends in the data using a weighted average of past observations.
•	Triple Exponential Smoothing: Extends double exponential smoothing by incorporating seasonal components.

![alt text](https://github.com/lavahammer/Adaptive-Stock-Predictor/blob/main/Adaptive%20Stock%20Predictor%20Data%20Architecture.png)
