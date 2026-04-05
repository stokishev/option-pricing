from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class OptionSetup:
    """
    A strict data structure to hold all parameters needed to price an option.
    """
    S: float      
    K: float      
    T: float      
    r: float      
    sigma: float  
    q: float = 0.0
    option_type: str = 'call'  

    def __post_init__(self):
        self.option_type = self.option_type.lower()
        if self.option_type not in ['call', 'put']:
            raise ValueError("option_type must be either 'call' or 'put'")

class PricingModel(ABC):
    """
    Abstract Base Class for all pricing models.
    Any class inheriting from this MUST implement the methods below.
    """
    def __init__(self, setup: OptionSetup):
        self.setup = setup

    @abstractmethod
    def calculate_price(self) -> float:
        """Calculates the theoretical price of the option."""
        pass

    @abstractmethod
    def calculate_greeks(self) -> dict:
        """Calculates the risk parameters: Delta, Gamma, Theta, Vega, Rho."""
        pass