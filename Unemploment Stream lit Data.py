
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import pickle

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv("Unemployment in India.csv")
    data.columns = data.columns.str.strip()
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data.dropna(subset=['Date'], inplace=True)
    data['Day'] = data['Date'].dt.day
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    return data

data = load_data()

# Sidebar title and filtering options
st.sidebar.title("Unemployment Analysis Dashboard")
st.sidebar.write("Explore unemployment data in India with visualizations and forecasts.")

# Main dashboard title
st.title("Unemployment Rate Analysis and Forecast in India")
st.write("Analyzing and forecasting unemployment trends using ARIMA.")

# Display data table on request
if st.sidebar.checkbox("Show Data"):
    st.write("Dataset Preview:")
    st.write(data.head())

# Plot Unemployment Rate Over Time
st.subheader("Unemployment Rate Over Time in India")
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='Date', y='Estimated Unemployment Rate (%)', marker='o', color='blue')
plt.title('Unemployment Rate Over Time in India')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)
st.pyplot(plt.gcf())

# Plot Unemployment Rate by Region
st.subheader("Unemployment Rate by Region")
plt.figure(figsize=(12, 6))
sns.barplot(data=data, x='Region', y='Estimated Unemployment Rate (%)', palette='Set2')
plt.title('Unemployment Rate by Region')
plt.xticks(rotation=45)
st.pyplot(plt.gcf())

# Histogram of Unemployment Rate Distribution
st.subheader("Distribution of Unemployment Rate")
plt.figure(figsize=(8, 5))
sns.histplot(data['Estimated Unemployment Rate (%)'], bins=15, kde=True, color='green')
plt.title('Distribution of Unemployment Rate')
plt.xlabel('Unemployment Rate (%)')
plt.ylabel('Frequency')
st.pyplot(plt.gcf())

# Scatter Plot: Month vs Unemployment Rate
st.subheader("Monthly Unemployment Rate Scatter Plot")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Month', y='Estimated Unemployment Rate (%)', hue='Year', palette='coolwarm')
plt.title('Unemployment Rate Scatter Plot (Month vs Year)')
plt.xlabel('Month')
plt.ylabel('Unemployment Rate (%)')
st.pyplot(plt.gcf())

# ARIMA Forecasting Section
st.subheader("Unemployment Rate Forecasting with ARIMA Model")

# Grouping the data by month for forecast
monthly_data = data.groupby('Month').agg({'Estimated Unemployment Rate (%)': 'mean'})

# Train-test split for forecasting
train_size = int(len(monthly_data) * 0.8)
train, test = monthly_data.iloc[:train_size], monthly_data.iloc[train_size:]

# Fit ARIMA model and forecast
model = ARIMA(train['Estimated Unemployment Rate (%)'], order=(1, 1, 1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=len(test))

# Calculate RMSE and Accuracy (MAPE)
rmse = np.sqrt(mean_squared_error(test['Estimated Unemployment Rate (%)'], forecast))
mape = np.mean(np.abs((test['Estimated Unemployment Rate (%)'] - forecast) / test['Estimated Unemployment Rate (%)'])) * 100
accuracy = 100 - mape
st.write(f"**RMSE**: {rmse:.2f}")
st.write(f"**Forecasting Accuracy**: {accuracy:.2f}%")

# Plot Actual vs Forecasted Unemployment Rate
st.subheader("Forecast vs Actual Unemployment Rate")
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Estimated Unemployment Rate (%)'], label='Training Data', color='blue')
plt.plot(test.index, test['Estimated Unemployment Rate (%)'], label='Actual Data', color='green')
plt.plot(test.index, forecast, label='Forecasted Data', color='red', linestyle='--')
plt.axvline(x=train.index[-1], color='gray', linestyle='--', label='Forecast Start')
plt.title('Unemployment Rate Forecast vs Actual')
plt.xlabel('Month')
plt.ylabel('Unemployment Rate (%)')
plt.legend()
plt.xticks(rotation=45)
st.pyplot(plt.gcf())
