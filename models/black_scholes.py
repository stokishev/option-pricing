import numpy as np
from scipy.stats import norm
from .base import PricingModel, OptionSetup

class BlackScholes(PricingModel):
    
    def _d1(self) -> float:
        return (np.log(self.setup.S / self.setup.K) + 
               (self.setup.r - self.setup.q + 0.5 * self.setup.sigma ** 2) * self.setup.T) / \
               (self.setup.sigma * np.sqrt(self.setup.T))
               
    def _d2(self) -> float:
        """Calculates the d2 probability factor."""
        return self._d1() - self.setup.sigma * np.sqrt(self.setup.T)

    def calculate_price(self) -> float:
        d1 = self._d1()
        d2 = d1 - self.setup.sigma * np.sqrt(self.setup.T)
        
        # Merton Extension for dividends
        if self.setup.option_type == 'call':
            price = (self.setup.S * np.exp(-self.setup.q * self.setup.T) * norm.cdf(d1) - 
                     self.setup.K * np.exp(-self.setup.r * self.setup.T) * norm.cdf(d2))
        else:
            price = (self.setup.K * np.exp(-self.setup.r * self.setup.T) * norm.cdf(-d2) - 
                     self.setup.S * np.exp(-self.setup.q * self.setup.T) * norm.cdf(-d1))
        return float(price)

    def calculate_greeks(self) -> dict:
        d1 = self._d1()
        d2 = self._d2()
        
        # Helper variables
        pdf_d1 = norm.pdf(d1)
        S = self.setup.S
        K = self.setup.K
        T = self.setup.T
        r = self.setup.r
        sigma = self.setup.sigma
        
        gamma = pdf_d1 / (S * sigma * np.sqrt(T))
        vega = S * pdf_d1 * np.sqrt(T) / 100  
        
        if self.setup.option_type == 'call':
            delta = norm.cdf(d1)
            theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            rho = (K * T * np.exp(-r * T) * norm.cdf(d2)) / 100
        else:
            delta = norm.cdf(d1) - 1
            theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            rho = (-K * T * np.exp(-r * T) * norm.cdf(-d2)) / 100
            
        return {
            "Delta": delta,
            "Gamma": gamma,
            "Theta": theta,
            "Vega": vega,
            "Rho": rho
        }