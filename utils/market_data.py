import yfinance as yf
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class MarketData:
    """Structured container for fetched market data."""
    ticker_symbol: str
    spot_price: float
    historical_volatility: float  # Annualized
    history: pd.DataFrame

class DataFetcher:
    """
    Handles fetching and cleaning data from Yahoo Finance.
    Keeps API logic isolated from the Streamlit UI.
    """
    @staticmethod
    def get_market_data(ticker_symbol: str, period: str = "1y") -> Optional[MarketData]:
        """
        Fetches historical data, spot price, and calculates annualized historical volatility.
        Returns None if the ticker is invalid or data cannot be fetched.
        """
        try:
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                return None
                
            # The most recent closing price is our current Spot Price (S)
            spot_price = float(hist['Close'].iloc[-1])
            
            # Calculate daily logarithmic returns
            hist['Log_Returns'] = np.log(hist['Close'] / hist['Close'].shift(1))
            
            # Calculate annualized historical volatility (assuming 252!!! trading days)
            daily_volatility = hist['Log_Returns'].std()
            annualized_vol = float(daily_volatility * np.sqrt(252))
            
            return MarketData(
                ticker_symbol=ticker_symbol.upper(),
                spot_price=spot_price,
                historical_volatility=annualized_vol,
                history=hist
            )
            
        except Exception as e:
            print(f"Failed to fetch data for {ticker_symbol}: {e}")
            return None