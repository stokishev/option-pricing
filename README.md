# 📈 Equity Derivatives Structuring Toolkit

A quantitative finance dashboard built for pricing, analyzing, and structuring equity derivatives. This toolkit implements Object-Oriented quantitative models to price European, American, and Exotic options, incorporating real-world market frictions and real-time market data pipelines.

## 🚀 Live Demo
**https://stokishev-option-pricing.streamlit.app/**

## ⚙️ Core Quantitative Features
* **Multi-Model Pricing Engine:** * **Black-Scholes-Merton:** Closed-form European pricing, extended to account for continuous dividend yields and borrow/repo costs.
  * **Cox-Ross-Rubinstein (Binomial Tree):** Vectorized backward induction for American early-exercise options.
  * **Monte Carlo Simulations:** Geometric Brownian Motion (GBM) engine for path-dependent Exotics (Asian Arithmetic Average, Up-and-Out Barriers with continuous path monitoring).
* **Practical Market Frictions:** Integrates dividend yields ($q$), borrow costs, and strict Actual/365 day-count conventions for time to maturity calculations.
* **Risk Management (The Greeks):** Calculates Delta, Gamma, Theta, Vega, and Rho using both closed-form partial derivatives and Finite Difference methods (Bump and Revalue).
* **Real-Time 3D Volatility Surface:** Fetches live options chain data and utilizes SciPy meshgrid interpolation to construct and visualize the Volatility Smile and Term Structure in 3D.
* **Automated Testing:** Fully unit-tested using `pytest` to benchmark against John Hull's textbook standards, verify Put-Call Parity, and ensure multi-model convergence.

## 💻 Tech Stack & Architecture
* **Frontend:** Streamlit, Plotly (Interactive visual analytics)
* **Quantitative Backend:** Python, NumPy (Vectorized path generation), SciPy (Statistical distributions, Grid interpolation)
* **Data Pipeline:** `yfinance`, Pandas
* **Architecture:** Strict OOP principles, Abstract Base Classes (ABCs), and `@dataclass` schemas for strict type-hinting and maintainable state management.

## 🛠️ Local Installation

```bash
# Clone the repository
git clone [https://github.com/stokishev/option-pricing.git](https://github.com/stokishev/option-pricing.git)
cd option-pricing

# Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate  # On Mac/Linux
pip install -r requirements.txt

# Run the test suite
python -m pytest tests/

# Launch the Streamlit application
streamlit run app.py