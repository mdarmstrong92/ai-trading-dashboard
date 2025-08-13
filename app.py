# working_dashboard.py - Complete Working AI Trading Dashboard

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="AI Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# Helper functions for technical indicators
def calculate_rsi(prices, window=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    exp1 = prices.ewm(span=fast).mean()
    exp2 = prices.ewm(span=slow).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, sma, lower

# Main dashboard
def main():
    st.title("🚀 AI Trading Dashboard")
    st.markdown("Real-time market analysis with technical indicators")
    
    # Sidebar
    st.sidebar.title("📊 Controls")
    
    # Stock selection
    default_stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    selected_stocks = st.sidebar.multiselect(
        "Select Stocks",
        options=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX'],
        default=default_stocks[:3]
    )
    
    # Time period
    period_options = {
        '1 Month': '1mo',
        '3 Months': '3mo', 
        '6 Months': '6mo',
        '1 Year': '1y'
    }
    selected_period = st.sidebar.selectbox(
        "Time Period",
        options=list(period_options.keys()),
        index=1
    )
    
    if not selected_stocks:
        st.warning("Please select at least one stock from the sidebar.")
        return
    
    # Fetch data
    with st.spinner("Fetching market data..."):
        stock_data = {}
        for symbol in selected_stocks:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period_options[selected_period])
                if not data.empty:
                    stock_data[symbol] = data
            except:
                st.error(f"Failed to fetch data for {symbol}")
    
    if not stock_data:
        st.error("No data available. Please try again.")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Technical Analysis", "⚡ Signals", "🔮 Predictions"])
    
    # Tab 1: Overview
    with tab1:
        st.header("📊 Market Overview")
        
        # Current prices
        cols = st.columns(len(selected_stocks))
        for i, symbol in enumerate(selected_stocks):
            if symbol in stock_data:
                data = stock_data[symbol]
                current_price = data['Close'].iloc[-1]
                prev_price = data['Close'].iloc[-2]
                change = current_price - prev_price
                change_pct = (change / prev_price) * 100
                
                with cols[i]:
                    st.metric(
                        label=symbol,
                        value=f"${current_price:.2f}",
                        delta=f"{change_pct:+.2f}%"
                    )
        
        # Price charts
        for symbol in selected_stocks:
            if symbol in stock_data:
                data = stock_data[symbol]
                
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name=symbol
                ))
                
                fig.update_layout(
                    title=f"{symbol} - Price Chart",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Technical Analysis
    with tab2:
        st.header("📈 Technical Analysis")
        
        selected_symbol = st.selectbox("Select Symbol for Analysis", selected_stocks)
        
        if selected_symbol in stock_data:
            data = stock_data[selected_symbol]
            
            # Calculate indicators
            data['SMA_20'] = data['Close'].rolling(20).mean()
            data['SMA_50'] = data['Close'].rolling(50).mean()
            data['RSI'] = calculate_rsi(data['Close'])
            
            macd, signal, histogram = calculate_macd(data['Close'])
            data['MACD'] = macd
            data['MACD_Signal'] = signal
            data['MACD_Histogram'] = histogram
            
            bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(data['Close'])
            data['BB_Upper'] = bb_upper
            data['BB_Middle'] = bb_middle
            data['BB_Lower'] = bb_lower
            
            # Create subplots
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=[
                    f"{selected_symbol} - Price & Moving Averages",
                    "RSI",
                    "MACD", 
                    "Bollinger Bands"
                ],
                vertical_spacing=0.08,
                row_heights=[0.4, 0.2, 0.2, 0.2]
            )
            
            # Price and MAs
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close', line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20', line=dict(color='red')), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50', line=dict(color='green')), row=1, col=1)
            
            # RSI
            fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            # MACD
            fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')), row=3, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal', line=dict(color='red')), row=3, col=1)
            fig.add_trace(go.Bar(x=data.index, y=data['MACD_Histogram'], name='Histogram'), row=3, col=1)
            
            # Bollinger Bands
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close', line=dict(color='blue')), row=4, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], name='BB Upper', line=dict(color='red')), row=4, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['BB_Middle'], name='BB Middle', line=dict(color='yellow')), row=4, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], name='BB Lower', line=dict(color='green')), row=4, col=1)
            
            fig.update_layout(height=1200, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Current values
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_rsi = data['RSI'].iloc[-1]
                rsi_signal = "SELL" if current_rsi > 70 else "BUY" if current_rsi < 30 else "HOLD"
                st.metric("RSI", f"{current_rsi:.1f}", rsi_signal)
            
            with col2:
                current_macd = data['MACD'].iloc[-1]
                current_signal = data['MACD_Signal'].iloc[-1]
                macd_signal = "BUY" if current_macd > current_signal else "SELL"
                st.metric("MACD", f"{current_macd:.4f}", macd_signal)
            
            with col3:
                current_price = data['Close'].iloc[-1]
                sma_20 = data['SMA_20'].iloc[-1]
                sma_signal = "BUY" if current_price > sma_20 else "SELL"
                st.metric("vs SMA 20", f"${sma_20:.2f}", sma_signal)
            
            with col4:
                bb_upper = data['BB_Upper'].iloc[-1]
                bb_lower = data['BB_Lower'].iloc[-1]
                if current_price > bb_upper:
                    bb_signal = "OVERBOUGHT"
                elif current_price < bb_lower:
                    bb_signal = "OVERSOLD"
                else:
                    bb_signal = "NEUTRAL"
                st.metric("Bollinger Bands", bb_signal, f"${current_price:.2f}")
    
    # Tab 3: Trading Signals
    with tab3:
        st.header("⚡ Trading Signals")
        
        signals_data = []
        
        for symbol in selected_stocks:
            if symbol in stock_data:
                data = stock_data[symbol]
                
                # Calculate indicators
                current_price = data['Close'].iloc[-1]
                rsi = calculate_rsi(data['Close']).iloc[-1]
                sma_20 = data['Close'].rolling(20).mean().iloc[-1]
                sma_50 = data['Close'].rolling(50).mean().iloc[-1]
                
                # Generate signals
                signal_score = 0
                signals = []
                
                if rsi < 30:
                    signal_score += 1
                    signals.append("RSI Oversold")
                elif rsi > 70:
                    signal_score -= 1
                    signals.append("RSI Overbought")
                
                if current_price > sma_20 > sma_50:
                    signal_score += 1
                    signals.append("Uptrend")
                elif current_price < sma_20 < sma_50:
                    signal_score -= 1
                    signals.append("Downtrend")
                
                # Overall signal
                if signal_score >= 1:
                    overall_signal = "🟢 BUY"
                elif signal_score <= -1:
                    overall_signal = "🔴 SELL"
                else:
                    overall_signal = "🟡 HOLD"
                
                signals_data.append({
                    'Symbol': symbol,
                    'Price': f"${current_price:.2f}",
                    'RSI': f"{rsi:.1f}",
                    'Signal Score': signal_score,
                    'Active Signals': ", ".join(signals) if signals else "None",
                    'Overall Signal': overall_signal
                })
        
        if signals_data:
            df = pd.DataFrame(signals_data)
            st.dataframe(df, use_container_width=True)
    
    # Tab 4: Simple Predictions
    with tab4:
        st.header("🔮 Simple Price Predictions")
        
        selected_symbol = st.selectbox("Select Symbol for Prediction", selected_stocks, key="pred_symbol")
        
        if selected_symbol in stock_data:
            data = stock_data[selected_symbol]
            
            # Simple moving average prediction
            current_price = data['Close'].iloc[-1]
            sma_5 = data['Close'].rolling(5).mean().iloc[-1]
            sma_20 = data['Close'].rolling(20).mean().iloc[-1]
            
            # Simple momentum calculation
            momentum = (current_price - data['Close'].iloc[-5]) / data['Close'].iloc[-5]
            predicted_change = momentum * 0.5  # Conservative prediction
            predicted_price = current_price * (1 + predicted_change)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
            
            with col2:
                change_pct = predicted_change * 100
                st.metric("Predicted Price (5 days)", f"${predicted_price:.2f}", f"{change_pct:+.2f}%")
            
            with col3:
                confidence = min(80, max(20, 60 - abs(momentum * 100)))
                st.metric("Confidence", f"{confidence:.0f}%")
            
            # Simple chart
            future_date = data.index[-1] + timedelta(days=5)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index[-30:],
                y=data['Close'].iloc[-30:],
                mode='lines',
                name='Historical Price'
            ))
            
            fig.add_trace(go.Scatter(
                x=[data.index[-1], future_date],
                y=[current_price, predicted_price],
                mode='lines+markers',
                name='Prediction',
                line=dict(dash='dash', color='red')
            ))
            
            fig.update_layout(
                title=f"{selected_symbol} - Simple Price Prediction",
                xaxis_title="Date",
                yaxis_title="Price ($)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("📝 This is a simple prediction based on recent momentum. For more accurate predictions, advanced ML models would be needed.")

if __name__ == "__main__":
    main()