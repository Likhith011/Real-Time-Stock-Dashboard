import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from sklearn.linear_model import LinearRegression

# Streamlit page config
st.set_page_config(page_title="📊 Real-Time Stock Dashboard", layout="wide")
st.title("📈 Real-Time Stock Market Dashboard")

# Sidebar settings
st.sidebar.header("Dashboard Controls")
symbols_input = st.sidebar.text_input("Enter stock tickers (comma-separated)", value="AAPL, MSFT")
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=60))
end_date = st.sidebar.date_input("End Date", datetime.now())

# Toggles
show_volume = st.sidebar.checkbox("Show Volume Chart", value=True)
show_sma20 = st.sidebar.checkbox("Show SMA (20)", value=True)
show_sma50 = st.sidebar.checkbox("Show SMA (50)", value=False)
show_ema20 = st.sidebar.checkbox("Show EMA (20)", value=True)
show_rsi = st.sidebar.checkbox("Show RSI (14)", value=False)
show_macd = st.sidebar.checkbox("Show MACD", value=False)
show_forecast = st.sidebar.checkbox("Show Price Forecast (Linear Regression)", value=False)
forecast_days = st.sidebar.slider("Forecast Days", 5, 30, 7) if show_forecast else None
auto_refresh = st.sidebar.checkbox("Enable Auto Refresh")
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 300, 60) if auto_refresh else None

# Helpers
@st.cache_data(ttl=300)
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    df.reset_index(inplace=True)
    return df

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = np.where(delta > 0, delta, 0).flatten()  # Ensure 1D
    loss = np.where(delta < 0, -delta, 0).flatten() # Ensure 1D
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data):
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def forecast_prices(df, days):
    df = df.copy()
    df['Day'] = range(len(df))
    model = LinearRegression()
    model.fit(df[['Day']], df['Close'])
    future_days = list(range(len(df), len(df) + days))
    future_prices = model.predict([[d] for d in future_days])
    future_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=days)
    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': np.array(future_prices).ravel()})
    return forecast_df

# Chart Drawing
def draw_charts(refresh_id=""):
    for symbol in symbols:
        df = load_data(symbol, start_date, end_date)
        if df.empty:
            st.warning(f"No data found for {symbol}")
            continue

        if show_sma20:
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
        if show_sma50:
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
        if show_ema20:
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        if show_rsi:
            df['RSI'] = calculate_rsi(df)
        if show_macd:
            df['MACD'], df['Signal'] = calculate_macd(df)

        st.subheader(f"{symbol} Stock Data")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price'))
        if show_sma20:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], mode='lines', name='SMA 20', line=dict(dash='dash')))
        if show_sma50:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], mode='lines', name='SMA 50', line=dict(dash='dot')))
        if show_ema20:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_20'], mode='lines', name='EMA 20', line=dict(dash='dashdot')))

        if show_forecast:
            forecast_df = forecast_prices(df, forecast_days)
            fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], mode='lines', name='Forecast', line=dict(color='red', dash='dot')))

        fig.update_layout(title=f"{symbol} Price Chart", xaxis_title="Date", yaxis_title="Price (USD)", height=450)
        st.plotly_chart(fig, use_container_width=True, key=f"{symbol}_price_{refresh_id}")

        if show_volume:
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume'))
            fig_vol.update_layout(title=f"{symbol} Volume", xaxis_title="Date", yaxis_title="Volume", height=300)
            st.plotly_chart(fig_vol, use_container_width=True, key=f"{symbol}_volume_{refresh_id}")
        if show_rsi:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI', line=dict(color='orange')))
            fig_rsi.update_layout(title=f"{symbol} RSI (14)", xaxis_title="Date", yaxis_title="RSI", yaxis_range=[0, 100], height=300)
            st.plotly_chart(fig_rsi, use_container_width=True, key=f"{symbol}_rsi_{refresh_id}")
        if show_macd:
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name='MACD', line=dict(color='cyan')))
            fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['Signal'], name='Signal', line=dict(color='magenta')))
            fig_macd.update_layout(title=f"{symbol} MACD & Signal", xaxis_title="Date", yaxis_title="Value", height=300)
            st.plotly_chart(fig_macd, use_container_width=True, key=f"{symbol}_macd_{refresh_id}")

        if show_forecast:
            st.markdown(f"### {symbol} Forecasted Closing Prices (Next {forecast_days} Days)")
            st.dataframe(forecast_df, use_container_width=True)

        st.markdown(f"### {symbol} Recent Data")
        st.dataframe(df.tail(), use_container_width=True)
        csv = convert_df_to_csv(df)
        st.download_button(
            label=f"📥 Download {symbol} data as CSV",
            data=csv,
            file_name=f"{symbol}_stock_data.csv",
            mime='text/csv',
            use_container_width=True,
            key=f"{symbol}_download_{refresh_id}"
        )

# Main run
if auto_refresh:
    if 'last_refresh' not in st.session_state:
        st.session_state['last_refresh'] = time.time()
    st.info(f"Auto-refreshing every {refresh_interval} seconds...")
    draw_charts(refresh_id=int(st.session_state['last_refresh']))
    if time.time() - st.session_state['last_refresh'] > refresh_interval:
        st.session_state['last_refresh'] = time.time()
        st.rerun()
else:
    draw_charts()
