import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
from datetime import datetime

def generate_volatility_surface(ticker_symbol: str) -> go.Figure:
    """
    Constructs a 3D Volatility Surface. 
    Manual session caching is disabled to maintain compatibility with 
    yfinance's internal TLS fingerprinting (curl_cffi).
    """
    ticker = yf.Ticker(ticker_symbol)
    expirations = ticker.options
    
    if not expirations:
        raise ValueError(f"No options data available for {ticker_symbol}")

    # Limit to front 10 expirations for term structure stability
    expirations = expirations[:10]
    
    data = []
    today = datetime.today()

    for exp in expirations:
        try:
            chain = ticker.option_chain(exp)
            calls = chain.calls
            
            exp_date = datetime.strptime(exp, '%Y-%m-%d')
            dte = (exp_date - today).days
            if dte <= 0:
                continue
                
            calls['DTE'] = dte
            
            # Clean data for meaningful surface: IV between 1% and 300%
            calls = calls[(calls['impliedVolatility'] > 0.01) & (calls['impliedVolatility'] < 3.0)]
            
            data.append(calls[['strike', 'DTE', 'impliedVolatility']])
        except Exception:
            continue

    if not data:
        raise ValueError("Insufficient data retrieved for surface construction.")

    df = pd.concat(data, ignore_index=True)
    
    # Mesh Grid Interpolation
    strikes = np.linspace(df['strike'].min(), df['strike'].max(), 50)
    dtes = np.linspace(df['DTE'].min(), df['DTE'].max(), 50)
    X, Y = np.meshgrid(strikes, dtes)
    
    # Interpolation: Linear with nearest-neighbor fill for edges
    Z = griddata((df['strike'], df['DTE']), df['impliedVolatility'], (X, Y), method='linear')
    Z_nearest = griddata((df['strike'], df['DTE']), df['impliedVolatility'], (X, Y), method='nearest')
    Z[np.isnan(Z)] = Z_nearest[np.isnan(Z)]

    # 3D Plotly Surface
    fig = go.Figure(data=[go.Surface(
        z=Z, x=X, y=Y, 
        colorscale='Viridis',
        hovertemplate="Strike: $%{x:.2f}<br>DTE: %{y:.0f}<br>IV: %{z:.2%}<extra></extra>"
    )])
    
    fig.update_layout(
        title=f'{ticker_symbol.upper()} Implied Volatility Surface',
        scene=dict(
            xaxis_title='Strike ($)',
            yaxis_title='DTE (Days)',
            zaxis_title='IV',
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        template='plotly_dark'
    )
    
    return fig