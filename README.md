# BSM Pricing

A Python package for Black-Scholes-Merton (BSM) option pricing, including inverse futures, implied volatility calculations, and volatility curve generation, fully vectorized for scalar or array inputs.

## Installation

Install the package directly from GitHub:

```bash
pip install git+https://github.com/clivelamkk/bsm_pricing.git
```

To install a specific version or update to the latest version:

```bash
pip install --upgrade git+https://github.com/clivelamkk/bsm_pricing.git
```

## Prerequisites

- Python 3.8+
- Dependencies: `numpy`, `scipy`, `pandas`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```python
from bsm_pricing import (
    gen_inverse_fut_local,
    gen_inverse_fut_usd,
    bsm_Nd2,
    general_bsm,
    inverse_bsm_local,
    general_bsm_iv,
    transform_laevitas_df,
    gen_CurveIV
)

# Calculate inverse futures in local currency
result = gen_inverse_fut_local(spots=100, strike=100, greek_type=1)
# result: array([0.0, 0.0001, -2e-06])  # [Price, Delta, Gamma]

# Calculate inverse futures in USD
result = gen_inverse_fut_usd(spots=[100, 200], strike=100, greek_type=1)
# result: array([[0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
#                [1.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0]])

# Calculate BSM in-the-money probability
prob = bsm_Nd2(spots=100, vols=0.2, k=100, tau=1.0, r=0.05, dvdYield=0.0, isCall=True)
# prob: float between 0 and 1

# Calculate BSM option price and greeks
greeks = general_bsm(spots=100, vols=0.2, k=100, tau=1.0, r=0.05, is_call=True, ccr=0.0, greek_type=1)
# greeks: array([price, delta, gamma, theta, vega, rho, vanna])

# Calculate inverse BSM in local currency
inverse_greeks = inverse_bsm_local(spots=100, vols=0.2, k=100, tau=1.0, r=0.05, is_call=True, ccr=0.0)
# inverse_greeks: array([price, delta, gamma, theta, vega, rho, vanna])

# Calculate implied volatility
price = general_bsm(100, 0.2, 100, 1.0, 0.05, True, 0.0, greek_type=0)
iv = general_bsm_iv(price, 100, 100, 1.0, 0.05, True, 0.0)
# iv: approximately 0.2

# Transform Laevitas API data for volatility curve generation
import pandas as pd
# Example input DataFrame (replace with actual data)
data = {
    'date': [1625097600000],  # Unix timestamp in ms
    'expiration_date': ['2022-12-30'],
    'underlyer': ['BTC'],
    'exercise': ['european'],
    'settlement': ['cash'],
    'strike': [35000],
    'forward_price': [34000],
    'claim_type': ['call'],
    'best_bid_price': [0.05],
    'best_ask_price': [0.06],
    'volume': [100],
    'open_interest': [500],
    'best_bid_amount': [10],
    'best_ask_amount': [12]
}
combined_df = pd.DataFrame(data)
transformed_df = transform_laevitas_df(combined_df)

# Generate implied volatility curves
isInverseQuoted = True
curve_df = gen_CurveIV(transformed_df, isInverseQuoted)
# curve_df: DataFrame with columns including 'underlying', 'date', 'expiry', 'strike', 'spot',
#           'bid_vol', 'ask_vol', 'bid_var', 'ask_var', 'moneyness'
```

## Functions

- `validate_inputs(*args)`: Validates that all array-like inputs have the same length.
- `broadcast_inputs(*args)`: Converts inputs to arrays, broadcasting scalars.
- `gen_inverse_fut_local(spots, strike, greek_type=1)`: Calculates inverse futures in local currency (BTC per 1USD).
- `gen_inverse_fut_usd(spots, strike, greek_type=1)`: Calculates inverse futures in USD.
- `bsm_Nd2(spots, vols, k, tau, r, dvdYield, isCall)`: Calculates the probability of being in-the-money (N(d2)).
- `general_bsm(spots, vols, k, tau, r, is_call, ccr, greek_type=1)`: Computes BSM option prices and greeks.
- `inverse_bsm_local(spots, vols, k, tau, r, is_call, ccr, greek_type=1)`: Computes BSM greeks in local currency using FX put-call parity.
- `general_bsm_iv(prices, spots, k, tau, r, is_call, ccr, n=10, m=10, cap_vol_at_max=False)`: Calculates implied volatility using a grid-based approach.
- `transform_laevitas_df(df)`: Transforms Laevitas API data into a format compatible with `gen_CurveIV`.
- `gen_CurveIV(df, isInverseQuoted)`: Generates implied volatility curves from options data, including bid/ask volatilities, variances, and moneyness.

## Dependencies

- numpy&gt;=1.21.0
- scipy&gt;=1.7.0
- pandas&gt;=1.4.0

## Development

To contribute or modify the package:

1. Clone the repository:

   ```bash
   git clone https://github.com/clivelamkk/bsm_pricing.git
   cd bsm_pricing
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run tests:

   ```bash
   pytest
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
