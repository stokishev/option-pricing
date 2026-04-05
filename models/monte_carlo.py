import numpy as np
import copy
from .base import PricingModel, OptionSetup

class MonteCarlo(PricingModel):
    """
    Monte Carlo Simulation Engine for Option Pricing.
    Supports European, Asian (Arithmetic Average), and Up-and-Out Barrier options.
    """
    def __init__(self, setup: OptionSetup, num_simulations: int = 10000, num_steps: int = 100, 
                 seed: int = 42, exotic_type: str = 'european', barrier: float = None):
        super().__init__(setup)
        self.num_simulations = num_simulations
        self.num_steps = num_steps
        self.seed = seed
        self.exotic_type = exotic_type.lower()
        self.barrier = barrier
        self.dt = self.setup.T / self.num_steps
        self.simulated_paths = None

    def generate_paths(self) -> np.ndarray:
        np.random.seed(self.seed)
        Z = np.random.standard_normal((self.num_simulations, self.num_steps))
        S = np.zeros((self.num_simulations, self.num_steps + 1))
        S[:, 0] = self.setup.S
        
        drift = (self.setup.r - 0.5 * self.setup.sigma ** 2) * self.dt
        
        for t in range(1, self.num_steps + 1):
            shock = self.setup.sigma * np.sqrt(self.dt) * Z[:, t-1]
            S[:, t] = S[:, t-1] * np.exp(drift + shock)
            
        self.simulated_paths = S
        return S

    def calculate_price(self) -> float:
        if self.simulated_paths is None:
            self.generate_paths()
            
        # Path-Dependent Logic
        if self.exotic_type == 'asian':
            # Payoff based on the average price of the path
            settlement_prices = np.mean(self.simulated_paths, axis=1)
        else:
            # European and Barrier base their payoff on the terminal price
            settlement_prices = self.simulated_paths[:, -1]
            
        # Calculate standard payoff
        if self.setup.option_type == 'call':
            payoffs = np.maximum(settlement_prices - self.setup.K, 0)
        else:
            payoffs = np.maximum(self.setup.K - settlement_prices, 0)
            
        # Apply Barrier Knock-Out Logic
        if self.exotic_type == 'up_and_out' and self.barrier is not None:
            # If any point in the path breached the barrier, the payoff becomes 0
            breach_mask = np.any(self.simulated_paths >= self.barrier, axis=1)
            payoffs[breach_mask] = 0.0
            
        discount_factor = np.exp(-self.setup.r * self.setup.T)
        expected_payoff = np.mean(payoffs)
        
        return float(discount_factor * expected_payoff)

    def calculate_greeks(self) -> dict:
        bump_size = 0.01 
        base_price = self.calculate_price()
        
        # Helper function to instantiate bumped models with identical exotic parameters
        def _get_bumped_mc(setup):
            return MonteCarlo(setup, self.num_simulations, self.num_steps, 
                              self.seed, self.exotic_type, self.barrier)
        
        # Delta
        setup_up = copy.deepcopy(self.setup)
        setup_up.S = self.setup.S * (1 + bump_size)
        price_up = _get_bumped_mc(setup_up).calculate_price()
        
        setup_down = copy.deepcopy(self.setup)
        setup_down.S = self.setup.S * (1 - bump_size)
        price_down = _get_bumped_mc(setup_down).calculate_price()
        
        delta = (price_up - price_down) / (2 * self.setup.S * bump_size)
        
        # Gamma
        gamma = (price_up - 2 * base_price + price_down) / ((self.setup.S * bump_size) ** 2)
        
        # Vega
        setup_vol = copy.deepcopy(self.setup)
        setup_vol.sigma = self.setup.sigma + 0.01 
        price_vol = _get_bumped_mc(setup_vol).calculate_price()
        vega = price_vol - base_price 

        return {"Delta": delta, "Gamma": gamma, "Theta": "N/A", "Vega": vega, "Rho": "N/A"}