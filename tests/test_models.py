import pytest
import numpy as np
from models.base import OptionSetup
from models.black_scholes import BlackScholes
from models.monte_carlo import MonteCarlo
from models.binomial_tree import BinomialTree

# Textbook Tests
def test_hull_textbook_black_scholes():
    """
    Tests against John Hull's textbook: S=42, K=40, r=10%, vol=20%, T=0.5 years.
    Expected Call: 4.76, Expected Put: 0.81
    """
    setup_call = OptionSetup(S=42.0, K=40.0, T=0.5, r=0.10, sigma=0.20, option_type='call')
    setup_put = OptionSetup(S=42.0, K=40.0, T=0.5, r=0.10, sigma=0.20, option_type='put')
    
    bs_call = BlackScholes(setup_call).calculate_price()
    bs_put = BlackScholes(setup_put).calculate_price()
    
    # pytest.approx allows for tiny floating-point rounding differences
    assert bs_call == pytest.approx(4.76, abs=0.01)
    assert bs_put == pytest.approx(0.81, abs=0.01)

# Put-Call Parity Tests
def test_put_call_parity_black_scholes():
    """
    Verifies that C - P = S - K * e^(-rT)
    """
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    setup_call = OptionSetup(S=S, K=K, T=T, r=r, sigma=sigma, option_type='call')
    setup_put = OptionSetup(S=S, K=K, T=T, r=r, sigma=sigma, option_type='put')
    
    C = BlackScholes(setup_call).calculate_price()
    P = BlackScholes(setup_put).calculate_price()
    
    lhs = C - P
    rhs = S - K * np.exp(-r * T)
    
    assert lhs == pytest.approx(rhs, rel=1e-4)

# Cross-Model Validation
def test_monte_carlo_convergence():
    """
    Tests if a high-simulation Monte Carlo prices a European option 
    identically to the closed-form Black-Scholes formula.
    """
    setup_call = OptionSetup(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, option_type='call')
    
    bs_price = BlackScholes(setup_call).calculate_price()
    
    # Use 100,000 simulations and a fixed seed to ensure a stable, accurate test
    mc_price = MonteCarlo(setup_call, num_simulations=100000, seed=42, exotic_type='european').calculate_price()
    
    # MC should be within 1% of the exact BS price
    assert mc_price == pytest.approx(bs_price, rel=0.01)

def test_binomial_tree_convergence():
    """
    Tests if a high-step Binomial Tree (forced to European execution) 
    converges to the exact Black-Scholes price.
    """
    setup_put = OptionSetup(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, option_type='put')
    
    bs_price = BlackScholes(setup_put).calculate_price()
    
    # Force is_american=False so it evaluates exactly like European Black-Scholes
    bt_price = BinomialTree(setup_put, num_steps=500, is_american=False).calculate_price()
    
    # BT should be extremely close to BS price
    assert bt_price == pytest.approx(bs_price, rel=0.005)