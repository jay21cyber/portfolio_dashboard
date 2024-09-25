import numpy as np
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# Function to get stock data
def get_stock_data(tickers, start, end):
    stock_data = {}
    for ticker in tickers:
        stock_data[ticker] = yf.download(ticker, start=start, end=end)['Adj Close']
    return pd.DataFrame(stock_data)

# Function to calculate portfolio metrics
def calculate_portfolio_metrics(df, weights):
    returns = df.pct_change().dropna()
    weights = np.array(weights)
    
    # Portfolio returns and volatility
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    
    # Assuming risk-free rate is 2%
    risk_free_rate = 0.07
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    return portfolio_return, portfolio_volatility, sharpe_ratio

# Streamlit UI
st.title("Portfolio Management Dashboard")

# Sidebar for user input
st.sidebar.header("Portfolio Allocation")

# Allow user to input stock tickers
tickers = st.sidebar.text_input("Enter stock tickers (comma-separated)", "AAPL,TSLA,MSFT").split(',')

# Allow user to input corresponding weights
weights_input = st.sidebar.text_input("Enter corresponding weights (comma-separated)", "0.4,0.3,0.3")
weights = [float(weight) for weight in weights_input.split(',')]

# Date input for start and end date
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Fetch stock data
df = get_stock_data(tickers, start_date, end_date)

# Display portfolio data
st.subheader("Portfolio Data")
st.write(df)

# Calculate metrics
portfolio_return, portfolio_volatility, sharpe_ratio = calculate_portfolio_metrics(df, weights)

# Display calculated metrics
st.subheader("Portfolio Performance Metrics")
st.write(f"Expected Annual Return: {portfolio_return:.2%}")
st.write(f"Annual Volatility: {portfolio_volatility:.2%}")
st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# Plot portfolio performance
st.subheader("Portfolio Performance Over Time")

# Create portfolio time series by calculating weighted sum of stock price changes
df['Portfolio'] = (df.pct_change().dropna() * weights).sum(axis=1).add(1).cumprod()

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Portfolio'], mode='lines', name='Portfolio'))
fig.update_layout(title="Portfolio Performance", xaxis_title="Date", yaxis_title="Portfolio Value", height=600)
st.plotly_chart(fig)
