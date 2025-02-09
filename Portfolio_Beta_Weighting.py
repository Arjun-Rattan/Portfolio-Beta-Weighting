import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress
import matplotlib.pyplot as plt

def get_historical_data(tickers, start_date, end_date):
    """Fetch historical closing prices for assets and benchmark"""
    data = yf.download(tickers + ['SPY'], start=start_date, end=end_date)
    if 'Adj Close' in data.columns:
        return data['Adj Close'].dropna()
    elif 'Close' in data.columns:
        return data['Close'].dropna()
    else:
        raise ValueError("Neither 'Adj Close' nor 'Close' columns found in the data")

def calculate_returns(data):
    """Calculate daily logarithmic returns"""
    return np.log(data / data.shift(1)).dropna()

def calculate_beta(asset_returns, benchmark_returns):
    """Calculate beta using linear regression"""
    slope, _, r_value, _, _ = linregress(benchmark_returns, asset_returns)
    return slope, r_value**2  # Beta and R-squared

def portfolio_beta_weighting(portfolio, total_portfolio_value, target_beta=1.0):
    """
    Calculate and adjust portfolio weights based on beta
    Args:
        portfolio (dict): {ticker: shares_owned}
        total_portfolio_value (float): Total portfolio value in USD
        target_beta (float): Desired portfolio beta
    """
    # Get current prices and positions
    data = yf.download(list(portfolio.keys()))
    if 'Adj Close' in data.columns:
        current_prices = data['Adj Close'].iloc[-1]
    elif 'Close' in data.columns:
        current_prices = data['Close'].iloc[-1]
    else:
        raise ValueError("Neither 'Adj Close' nor 'Close' columns found in the data")
    
    positions = {ticker: shares * current_prices[ticker] 
                 for ticker, shares in portfolio.items()}
    
    # Historical returns calculation
    data = get_historical_data(list(portfolio.keys()), '2020-01-01', '2023-01-01')
    returns = calculate_returns(data)
    spy_returns = returns['SPY']
    
    # Calculate individual betas
    betas = {}
    for ticker in portfolio:
        beta, r_sq = calculate_beta(returns[ticker], spy_returns)
        betas[ticker] = beta
    
    # Current portfolio beta
    current_weights = np.array(list(positions.values())) / total_portfolio_value
    current_beta = np.dot(list(betas.values()), current_weights)
    
    # Beta adjustment calculations
    beta_adjustment = target_beta / current_beta
    new_weights = current_weights * beta_adjustment
    normalized_weights = new_weights / new_weights.sum()
    
    # Generate output
    results = pd.DataFrame({
        'Ticker': list(portfolio.keys()),
        'Current Shares': list(portfolio.values()),
        'Current Price': [current_prices[t] for t in portfolio],
        'Beta': [betas[t] for t in portfolio],
        'Current Weight': current_weights,
        'Adjusted Weight': normalized_weights
    })
    
    # Calculate new shares needed
    results['New Shares'] = np.round(
        (normalized_weights * total_portfolio_value) / results['Current Price'], 0
    )
    
    return results

if __name__ == "__main__":
    # Example usage
    portfolio = {
        'AAPL': 100,   # Apple shares
        'MSFT': 50,    # Microsoft shares
        'AMZN': 30     # Amazon shares
    }
    portfolio_value = 1000000  # $1,000,000 total value
    
    results = portfolio_beta_weighting(portfolio, portfolio_value)
    
    print("\nPortfolio Beta Analysis:")
    print(results[['Ticker', 'Beta', 'Current Weight', 'Adjusted Weight', 'New Shares']])
    
    # Plot beta comparisons
    plt.figure(figsize=(10, 6))
    plt.bar(results['Ticker'], results['Beta'])
    plt.title('Asset Betas Relative to S&P 500')
    plt.ylabel('Beta')
    plt.axhline(1.0, color='r', linestyle='--', label='Market Beta (1.0)')
    plt.legend()
    plt.show()




