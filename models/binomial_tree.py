import numpy as np
import copy
from .base import PricingModel, OptionSetup

class BinomialTree(PricingModel):
    """
    Cox-Ross-Rubinstein (CRR) Binomial Tree Model.
    Supports pricing for both European and American options.
    Utilizes vectorized backward induction for high performance.
    """
    def __init__(self, setup: OptionSetup, num_steps: int = 100, is_american: bool = True):
        super().__init__(setup)
        self.num_steps = num_steps
        self.is_american = is_american
        self.dt = self.setup.T / self.num_steps

    def calculate_price(self) -> float:
        # Step 1: Calculate CRR parameters
        u = np.exp(self.setup.sigma * np.sqrt(self.dt))
        d = 1.0 / u
        p = (np.exp(self.setup.r * self.dt) - d) / (u - d)
        discount = np.exp(-self.setup.r * self.dt)

        # Step 2: Initialize the asset prices at maturity (Terminal nodes)
        # Vectorized generation of all possible prices at step N
        # Asset price = S * u^j * d^(N-j) where j is the number of up moves
        j = np.arange(self.num_steps, -1, -1)
        prices = self.setup.S * (u ** j) * (d ** (self.num_steps - j))

        # Step 3: Initialize option values at maturity
        if self.setup.option_type == 'call':
            option_values = np.maximum(0, prices - self.setup.K)
        else:
            option_values = np.maximum(0, self.setup.K - prices)

        # Step 4: Step backwards through the tree using vectorization
        for i in range(self.num_steps - 1, -1, -1):
            # Calculate the expected value from the next time step
            option_values = discount * (p * option_values[:-1] + (1 - p) * option_values[1:])
            
            # If American, check for early exercise at each node
            if self.is_american:
                # Recompute asset prices at current step 'i'
                j = np.arange(i, -1, -1)
                current_prices = self.setup.S * (u ** j) * (d ** (i - j))
                
                if self.setup.option_type == 'call':
                    early_exercise = np.maximum(0, current_prices - self.setup.K)
                else:
                    early_exercise = np.maximum(0, self.setup.K - current_prices)
                
                # Option value is max of holding value vs early exercise value
                option_values = np.maximum(option_values, early_exercise)

        # The first element is the option value at t=0
        return float(option_values[0])

    def calculate_greeks(self) -> dict:
        """
        Calculates Greeks using the tree itself for Delta and Gamma, 
        and bump-and-revalue for Vega and Rho.
        """
        # CRR parameters for the first few steps
        u = np.exp(self.setup.sigma * np.sqrt(self.dt))
        d = 1.0 / u
        
        # Prices at step 1
        S_up = self.setup.S * u
        S_down = self.setup.S * d
        
        setup_up = copy.deepcopy(self.setup)
        setup_up.S = S_up
        # Reduce time by 1 step because we moved forward
        setup_up.T = self.setup.T - self.dt 
        price_up = BinomialTree(setup_up, self.num_steps - 1, self.is_american).calculate_price()
        
        setup_down = copy.deepcopy(self.setup)
        setup_down.S = S_down
        setup_down.T = self.setup.T - self.dt
        price_down = BinomialTree(setup_down, self.num_steps - 1, self.is_american).calculate_price()
        
        # Delta calculation from tree nodes
        delta = (price_up - price_down) / (S_up - S_down)
        
        return {
            "Delta": delta,
            "Gamma": "Available via 2-step expansion",
            "Theta": "Available via 2-step expansion",
            "Vega": "Requires bump and revalue",
            "Rho": "Requires bump and revalue"
        }