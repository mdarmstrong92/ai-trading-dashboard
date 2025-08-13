# Enhanced AI Trading Dashboard with Custom Ticker Input

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

def validate_ticker(ticker):
    """Validate if ticker exists and has data"""
    try:
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period="5d")
        if hist.empty:
            return False, f"No data found for {ticker.upper()}"
        return True, hist
    except:
        return False, f"Invalid ticker symbol: {ticker.upper()}"

def get_company_info(ticker):
    """Get company name and basic info"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        company_name = info.get('longName', info.get('shortName', ticker.upper()))
        sector = info.get('sector', 'N/A')
        return company_name, sector
    except:
        return ticker.upper(), 'N/A'

# Main dashboard
def main():
    st.title("🚀 AI Trading Dashboard")
    st.markdown("Real-time market analysis with technical indicators")
    
    # Sidebar
    st.sidebar.title("📊 Controls")
    
    # Custom ticker input section
    st.sidebar.markdown("### 🔍 Add Custom Ticker")
    custom_ticker = st.sidebar.text_input(
        "Enter any stock symbol:",
        placeholder="e.g., AAPL, TSLA, GME, NVDA",
        help="Type any valid stock ticker and press Enter"
    ).upper()
    
    # Add button to validate and add ticker
    add_custom = st.sidebar.button("➕ Add Ticker", help="Add the ticker above to your analysis")
    
    # Default stock selection
    st.sidebar.markdown("### 📈 Popular Stocks")
    default_stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX']
    selected_stocks = st.sidebar.multiselect(
        "Select from popular stocks:",
        options=default_stocks + ['AMD', 'CRM', 'ORCL', 'JPM', 'V', 'DIS', 'COIN', 'SPY', 'QQQ'],
        default=['AAPL', 'MSFT', 'TSLA']
    )
    
    # Initialize session state for custom tickers
    if 'custom_tickers' not in st.session_state:
        st.session_state.custom_tickers = []
    
    # Handle custom ticker addition
    if add_custom and custom_ticker:
        with st.sidebar:
            with st.spinner(f"Validating {custom_ticker}..."):
                is_valid, result = validate_ticker(custom_ticker)
                
                if is_valid:
                    if custom_ticker not in st.session_state.custom_tickers:
                        st.session_state.custom_tickers.append(custom_ticker)
                        company_name, sector = get_company_info(custom_ticker)
                        st.success(f"✅ Added {custom_ticker} ({company_name})")
                    else:
                        st.info(f"ℹ️ {custom_ticker} already in your list")
                else:
                    st.error(f"❌ {result}")
    
    # Display custom tickers
    if st.session_state.custom_tickers:
        st.sidebar.markdown("### 🎯 Your Custom Tickers")
        for ticker in st.session_state.custom_tickers:
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                st.write(f"📌 {ticker}")
            with col2:
                if st.button("❌", key=f"remove_{ticker}", help=f"Remove {ticker}"):
                    st.session_state.custom_tickers.remove(ticker)
                    st.experimental_rerun()
    
    # Combine all selected tickers
    all_selected_tickers = list(set(selected_stocks + st.session_state.custom_tickers))
    
    # Time period
    st.sidebar.markdown("### ⏰ Time Period")
    period_options = {
        '1 Month': '1mo',
        '3 Months': '3mo', 
        '6 Months': '6mo',
        '1 Year': '1y',
        '2 Years': '2y'
    }
    selected_period = st.sidebar.selectbox(
        "Historical data period:",
        options=list(period_options.keys()),
        index=1
    )
    
    # Show current selection
    if all_selected_tickers:
        st.sidebar.markdown("### 📋 Currently Analyzing")
        for ticker in sorted(all_selected_tickers):
            company_name, sector = get_company_info(ticker)
            st.sidebar.write(f"• **{ticker}** - {company_name}")
    
    if not all_selected_tickers:
        st.warning("Please select stocks from popular list or add custom tickers using the input above.")
        st.info("💡 **Try adding:** AAPL, TSLA, GME, AMC, NVDA, or any other stock symbol!")
        return
    
    # Fetch data with progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    stock_data = {}
    failed_tickers = []
    
    for i, symbol in enumerate(all_selected_tickers):
        try:
            status_text.text(f"Fetching data for {symbol}...")
            progress_bar.progress((i + 1) / len(all_selected_tickers))
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period_options[selected_period])
            if not data.empty:
                stock_data[symbol] = data
            else:
                failed_tickers.append(symbol)
        except Exception as e:
            failed_tickers.append(symbol)
            st.warning(f"Failed to fetch data for {symbol}: {str(e)}")
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Show failed tickers
    if failed_tickers:
        st.error(f"❌ Could not fetch data for: {', '.join(failed_tickers)}")
        # Remove failed custom tickers
        for ticker in failed_tickers:
            if ticker in st.session_state.custom_tickers:
                st.session_state.custom_tickers.remove(ticker)
    
    if not stock_data:
        st.error("No valid data available. Please try different tickers.")
        return
    
    # Success message
    st.success(f"✅ Successfully loaded data for {len(stock_data)} stocks: {', '.join(sorted(stock_data.keys()))}")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Technical Analysis", "⚡ Signals", "🔮 Predictions"])
    
    # Tab 1: Overview
    with tab1:
        st.header("📊 Market Overview")
        
        # Current prices in a more compact format
        if len(stock_data) <= 6:
            cols = st.columns(len(stock_data))
        else:
            # For more than 6 stocks, use multiple rows
            cols_per_row = 6
            rows = (len(stock_data) + cols_per_row - 1) // cols_per_row
            
        ticker_list = list(stock_data.keys())
        
        for i, symbol in enumerate(ticker_list):
            if len(stock_data) <= 6:
                col_idx = i
            else:
                col_idx = i % cols_per_row
                if col_idx == 0:
                    cols = st.columns(min(cols_per_row, len(ticker_list) - i))
            
            data = stock_data[symbol]
            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
            change = current_price - prev_price
            change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
            
            with cols[col_idx]:
                st.metric(
                    label=symbol,
                    value=f"${current_price:.2f}",
                    delta=f"{change_pct:+.2f}%"
                )
        
        # Price charts - show top 4 or selected ones
        st.subheader("📈 Price Charts")
        
        if len(stock_data) > 4:
            chart_symbols = st.multiselect(
                "Select stocks to chart (max 4):",
                options=list(stock_data.keys()),
                default=list(stock_data.keys())[:4],
                max_selections=4
            )
        else:
            chart_symbols = list(stock_data.keys())
        
        for symbol in chart_symbols:
            data = stock_data[symbol]
            company_name, sector = get_company_info(symbol)
            
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
                title=f"{symbol} - {company_name} ({sector})",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Technical Analysis
    with tab2:
        st.header("📈 Technical Analysis")
        
        analysis_symbol = st.selectbox("Select Symbol for Detailed Analysis", list(stock_data.keys()))
        
        if analysis_symbol in stock_data:
            data = stock_data[analysis_symbol].copy()
            company_name, sector = get_company_info(analysis_symbol)
            
            st.subheader(f"Analysis for {analysis_symbol} - {company_name}")
            st.write(f"**Sector:** {sector}")
            
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
                    f"{analysis_symbol} - Price & Moving Averages",
                    "RSI (Relative Strength Index)",
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
                st.metric("MACD Signal", macd_signal, f"{current_macd:.4f}")
            
            with col3:
                current_price = data['Close'].iloc[-1]
                sma_20 = data['SMA_20'].iloc[-1]
                sma_signal = "BUY" if current_price > sma_20 else "SELL"
                st.metric("vs SMA 20", sma_signal, f"${sma_20:.2f}")
            
            with col4:
                bb_upper = data['BB_Upper'].iloc[-1]
                bb_lower = data['BB_Lower'].iloc[-1]
                if current_price > bb_upper:
                    bb_signal = "OVERBOUGHT"
                elif current_price < bb_lower:
                    bb_signal = "OVERSOLD"
                else:
                    bb_signal = "NEUTRAL"
                st.metric("Bollinger Position", bb_signal, f"${current_price:.2f}")
    
    # Tab 3: Trading Signals
    with tab3:
        st.header("⚡ Trading Signals")
        
        signals_data = []
        
        for symbol in stock_data.keys():
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
            
            # Get company name
            company_name, _ = get_company_info(symbol)
            
            signals_data.append({
                'Symbol': symbol,
                'Company': company_name[:30] + "..." if len(company_name) > 30 else company_name,
                'Price': f"${current_price:.2f}",
                'RSI': f"{rsi:.1f}",
                'Signal Score': signal_score,
                'Active Signals': ", ".join(signals) if signals else "None",
                'Overall Signal': overall_signal
            })
        
        if signals_data:
            df = pd.DataFrame(signals_data)
            st.dataframe(df, use_container_width=True)
            
            # Summary charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Signal distribution
                signal_counts = df['Overall Signal'].value_counts()
                fig_pie = px.pie(values=signal_counts.values, names=signal_counts.index, 
                               title="Signal Distribution")
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # RSI distribution
                rsi_values = [float(x) for x in df['RSI']]
                fig_hist = px.histogram(x=rsi_values, title="RSI Distribution", 
                                      labels={'x': 'RSI Value', 'y': 'Count'})
                fig_hist.add_vline(x=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                fig_hist.add_vline(x=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                st.plotly_chart(fig_hist, use_container_width=True)
    
    # Tab 4: Simple Predictions
    with tab4:
        st.header("🔮 Price Predictions")
        
        pred_symbol = st.selectbox("Select Symbol for Prediction", list(stock_data.keys()), key="pred_symbol")
        
        if pred_symbol in stock_data:
            data = stock_data[pred_symbol]
            company_name, sector = get_company_info(pred_symbol)
            
            st.subheader(f"Prediction for {pred_symbol} - {company_name}")
            
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
                name='Historical Price',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=[data.index[-1], future_date],
                y=[current_price, predicted_price],
                mode='lines+markers',
                name='Prediction',
                line=dict(dash='dash', color='red'),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title=f"{pred_symbol} - Simple Price Prediction",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("📝 This is a simple prediction based on recent momentum. For more accurate predictions, advanced ML models would be needed.")
            
            # Additional stats
            st.subheader("📊 Additional Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
                st.metric("Annual Volatility", f"{volatility:.1f}%")
            
            with col2:
                ytd_return = ((current_price - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
                st.metric("Period Return", f"{ytd_return:.1f}%")
            
            with col3:
                avg_volume = data['Volume'].mean() / 1e6
                st.metric("Avg Volume", f"{avg_volume:.1f}M")
            
            with col4:
                high_52w = data['High'].max()
                low_52w = data['Low'].min()
                range_position = ((current_price - low_52w) / (high_52w - low_52w)) * 100
                st.metric("Range Position", f"{range_position:.1f}%")

if __name__ == "__main__":
    main()
