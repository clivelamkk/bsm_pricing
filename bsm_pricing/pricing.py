import numpy as np
from scipy.stats import norm

def validate_inputs(*args):
    """Validate that all array-like inputs have the same length.
    
    Args:
        *args: Variable number of inputs, which can be scalars (int, float, bool),
            arrays (list, np.ndarray), or None. Arrays must have the same length.
            None values are ignored.
    
    Returns:
        int: Length of array inputs (or 1 if all inputs are scalars or None).
    
    Raises:
        ValueError: If array inputs have different lengths.
    """
    inputs = [x for x in args if isinstance(x, (list, np.ndarray)) and x is not None]
    if inputs:
        lengths = [len(x) for x in inputs]
        if len(set(lengths)) > 1:
            raise ValueError("All array inputs must have the same length")
        return lengths[0]
    return 1  # If all inputs are scalars or None, assume 1 scenario

def broadcast_inputs(*args):
    """Convert inputs to arrays of length n, broadcasting scalars.
    
    The length n is determined by calling validate_inputs on the arguments.
    
    Args:
        *args: Variable number of inputs, which can be scalars, arrays, or None.
    
    Returns:
        tuple: Arrays of length n for each input. None inputs become arrays of None.
    
    Raises:
        ValueError: If array inputs have different lengths.
    """
    n = validate_inputs(*args)
    
    def to_array(x, dtype=None):
        if isinstance(x, (list, np.ndarray)):
            return np.asarray(x, dtype=dtype)
        return np.full(n, x, dtype=dtype)
    
    return tuple(to_array(arg, dtype=bool if i == len(args)-1 and isinstance(arg, (bool, np.ndarray)) and (isinstance(arg, bool) or arg.dtype == bool) else None)
                 for i, arg in enumerate(args))

def gen_inverse_fut_local(spots, strike, greek_type=1):
    """Value of fut in BTC per 1USD size, vectorized, accepts scalar or array inputs."""
    spots, strike = broadcast_inputs(spots, strike)
    n = len(spots)  # Derive n from broadcasted array length
    
    if greek_type == 0:
        pv = np.zeros(n)
    else:
        pv = np.zeros((n, 3))

    if greek_type == 0:
        pv = 1 / strike - 1 / spots
    else:
        pv[:, 0] = 1 / strike - 1 / spots  # Price
        pv[:, 1] = 1 / (spots ** 2)  # Delta
        pv[:, 2] = -2 / (spots ** 3)  # Gamma

    return pv

def gen_inverse_fut_usd(spots, strike, greek_type=1):
    """Value of fut in USD per 1USD size, vectorized, accepts scalar or array inputs."""
    spots, strike = broadcast_inputs(spots, strike)
    n = len(spots)  # Derive n from broadcasted array length
    
    if greek_type == 0:
        pv = np.zeros(n)
    else:
        pv = np.zeros((n, 7 if greek_type == 1 else 13))

    if greek_type == 0:
        pv = spots / strike - 1
    else:
        pv[:, 0] = spots / strike - 1  # Price
        pv[:, 1] = 1 / strike  # Delta
        pv[:, 2:7] = 0  # Gamma, Theta, Vega, Rho, Vanna
        if greek_type > 1:
            pv[:, 7:13] = 0  # Charm, Vomma, Veta, Speed, Zomma, Ultima

    return pv

def bsm_Nd2(spots, vols, k, tau, r, dvdYield, isCall):
    """
    Calculate the probability of being in-the-money in the Black-Scholes formula, vectorized.
    Returns N(d2) for calls and N(-d2) = 1 - N(d2) for puts.
    
    Args:
        spots: Spot price(s) (scalar or array).
        vols: Volatility(s) (scalar or array).
        k: Strike price(s) (scalar or array).
        tau: Time to expiration(s) (scalar or array).
        r: Risk-free rate(s) (scalar or array).
        dvdYield: Continuous dividend yield(s) (scalar or array).
        isCall: Boolean, True for call, False for put (scalar or array).
    
    Returns:
        np.ndarray: In-the-money probabilities (N(d2) for calls, 1 - N(d2) for puts).
                   Returns 0 for invalid cases (e.g., spots <= 0, k <= 0, vols <= 0, tau <= 0).
    """
    # Broadcast inputs to ensure consistent array lengths
    spots, vols, k, tau, r, dvdYield, isCall = broadcast_inputs(spots, vols, k, tau, r, dvdYield, isCall)
    n = len(spots)  # Derive n from broadcasted array length
    
    # Initialize output array
    n_d2 = np.zeros(n)
    
    # Identify invalid cases
    spot_leq_0 = spots <= 0
    k_leq_0 = k <= 0
    vol_leq_0 = vols <= 0
    tau_leq_0 = tau <= 0
    normal_case = ~(spot_leq_0 | k_leq_0 | vol_leq_0 | tau_leq_0)
    
    # For invalid cases, N(d2) remains 0 (as initialized)
    
    # Compute N(d2) for normal cases
    if np.any(normal_case):
        sqr_tau = np.sqrt(tau[normal_case])
        # Compute d2 directly
        d2 = (np.log(spots[normal_case] / k[normal_case]) + 
              tau[normal_case] * (r[normal_case] - dvdYield[normal_case] - 0.5 * vols[normal_case]**2)) / (vols[normal_case] * sqr_tau)
        # Compute N(d2)
        n_d2[normal_case] = norm.cdf(d2)
        # Adjust for puts: N(-d2) = 1 - N(d2)
        put_mask = ~isCall[normal_case]
        n_d2[normal_case] = np.where(put_mask, 1 - n_d2[normal_case], n_d2[normal_case])
    
    return n_d2

def general_bsm(spots, vols, k, tau, r, is_call, ccr, greek_type=1):
    """Black-Scholes-Merton model, vectorized for separate spot and vol arrays."""
    spots, vols, k, tau, r, ccr, is_call = broadcast_inputs(spots, vols, k, tau, r, ccr, is_call)
    n = len(spots)  # Derive n from broadcasted array length
    
    if greek_type == 0:
        pv = np.zeros(n)
    elif greek_type == 1:
        pv = np.zeros((n, 7))
    else:
        pv = np.zeros((n, 13))

    price = np.zeros(n)
    delta = np.zeros(n)
    gamma = np.zeros(n)
    vega = np.zeros(n)
    theta = np.zeros(n)
    rho = np.zeros(n)
    vanna = np.zeros(n)
    charm = np.zeros(n)
    vomma = np.zeros(n)
    veta = np.zeros(n)
    speed = np.zeros(n)
    zomma = np.zeros(n)
    ultima = np.zeros(n)

    spot_leq_0 = spots <= 0
    k_leq_0 = k <= 0
    vol_leq_0 = vols <= 0
    tau_leq_0 = tau <= 0
    normal_case = ~(spot_leq_0 | k_leq_0 | vol_leq_0 | tau_leq_0)

    if np.any(spot_leq_0):
        not_call = ~is_call
        mask = spot_leq_0 & not_call
        price[mask] = k[mask] * np.exp(-r[mask] * tau[mask])
        delta[mask] = -np.exp((ccr[mask] - r[mask]) * tau[mask])

    if np.any(k_leq_0):
        mask = k_leq_0 & is_call
        price[mask] = spots[mask] * np.exp((ccr[mask] - r[mask]) * tau[mask])
        delta[mask] = np.exp((ccr[mask] - r[mask]) * tau[mask])

    if np.any(vol_leq_0):
        threshold = k * np.exp(ccr * tau)
        call_mask = is_call & vol_leq_0
        put_mask = (~is_call) & vol_leq_0
        above = (spots > threshold) & call_mask
        equal_call = (spots == threshold) & call_mask
        below = (spots < threshold) & put_mask
        equal_put = (spots == threshold) & put_mask

        price[above] = spots[above] * np.exp((ccr[above] - r[above]) * tau[above]) - k[above] * np.exp(-r[above] * tau[above])
        delta[above] = np.exp((ccr[above] - r[above]) * tau[above])
        gamma[equal_call] = 100

        price[below] = k[below] * np.exp(-r[below] * tau[below]) - spots[below] * np.exp((ccr[below] - r[below]) * tau[below])
        delta[below] = -np.exp((ccr[below] - r[below]) * tau[below])
        gamma[equal_put] = 100

    if np.any(tau_leq_0):
        call_mask = is_call & tau_leq_0
        put_mask = (~is_call) & tau_leq_0
        above = (spots > k) & call_mask
        below = (spots < k) & put_mask
        equal = (spots == k) & tau_leq_0

        price[above] = spots[above] - k[above]
        delta[above] = 1
        price[below] = k[below] - spots[below]
        delta[below] = -1
        gamma[equal] = 100
        gamma[~equal & tau_leq_0] = 0

    if np.any(normal_case):
        sqr_tau = np.sqrt(tau[normal_case])
        exp_bm_rt = np.exp((ccr[normal_case] - r[normal_case]) * tau[normal_case])
        d1 = (np.log(spots[normal_case] / k[normal_case]) + tau[normal_case] * (ccr[normal_case] + 0.5 * vols[normal_case]**2)) / (vols[normal_case] * sqr_tau)
        d2 = d1 - vols[normal_case] * sqr_tau
        nd1 = norm.cdf(d1)
        nd2 = norm.cdf(d2)
        nmd1 = 1 - nd1
        nmd2 = 1 - nd2
        n_d_d1 = norm.pdf(d1)

        call_mask = is_call[normal_case]
        put_mask = ~is_call[normal_case]

        if np.any(call_mask):
            c_idx = normal_case & is_call
            price[c_idx] = nd1[call_mask] * spots[c_idx] * exp_bm_rt[call_mask] - nd2[call_mask] * k[c_idx] * np.exp(-r[c_idx] * tau[c_idx])
            if greek_type > 0:
                delta[c_idx] = exp_bm_rt[call_mask] * nd1[call_mask]
                theta[c_idx] = (-spots[c_idx] * n_d_d1[call_mask] * vols[c_idx] * exp_bm_rt[call_mask] / (2 * sqr_tau[call_mask]) -
                                (ccr[c_idx] - r[c_idx]) * spots[c_idx] * exp_bm_rt[call_mask] * nd1[call_mask] -
                                r[c_idx] * k[c_idx] * np.exp(-r[c_idx] * tau[c_idx]) * nd2[call_mask])
                rho[c_idx] = k[c_idx] * tau[c_idx] * np.exp(-r[c_idx] * tau[c_idx]) * nd2[call_mask]

        if np.any(put_mask):
            p_idx = normal_case & ~is_call
            price[p_idx] = nmd2[put_mask] * k[p_idx] * np.exp(-r[p_idx] * tau[p_idx]) - nmd1[put_mask] * spots[p_idx] * exp_bm_rt[put_mask]
            if greek_type > 0:
                delta[p_idx] = -exp_bm_rt[put_mask] * nmd1[put_mask]
                theta[p_idx] = (-spots[p_idx] * n_d_d1[put_mask] * vols[p_idx] * exp_bm_rt[put_mask] / (2 * sqr_tau[put_mask]) +
                                (ccr[p_idx] - r[p_idx]) * spots[p_idx] * exp_bm_rt[put_mask] * nmd1[put_mask] +
                                r[p_idx] * k[p_idx] * np.exp(-r[p_idx] * tau[p_idx]) * nmd2[put_mask])
                rho[p_idx] = -k[p_idx] * tau[p_idx] * np.exp(-r[p_idx] * tau[p_idx]) * nmd2[put_mask]

        if greek_type > 0:
            gamma[normal_case] = n_d_d1 * exp_bm_rt / (spots[normal_case] * vols[normal_case] * sqr_tau)
            vega[normal_case] = spots[normal_case] * exp_bm_rt * n_d_d1 * sqr_tau
            vanna[normal_case] = -exp_bm_rt * n_d_d1 * d2 / vols[normal_case]

        if greek_type > 1:
            call_mask = is_call[normal_case]
            charm[normal_case & is_call] = (exp_bm_rt[call_mask] * n_d_d1[call_mask] * (ccr[normal_case & is_call] / (vols[normal_case & is_call] * sqr_tau[call_mask]) - d2[call_mask] / (2 * tau[normal_case & is_call])) -
                                           (ccr[normal_case & is_call] - r[normal_case & is_call]) * exp_bm_rt[call_mask] * nd1[call_mask])
            charm[normal_case & ~is_call] = (exp_bm_rt[put_mask] * n_d_d1[put_mask] * (ccr[normal_case & ~is_call] / (vols[normal_case & ~is_call] * sqr_tau[put_mask]) - d2[put_mask] / (2 * tau[normal_case & ~is_call])) +
                                            (ccr[normal_case & ~is_call] - r[normal_case & ~is_call]) * exp_bm_rt[put_mask] * nmd1[put_mask])
            vomma[normal_case] = vega[normal_case] * d1 * d2 / vols[normal_case]
            veta[normal_case] = spots[normal_case] * exp_bm_rt * n_d_d1 * sqr_tau * (
                -(ccr[normal_case] - r[normal_case]) + 0.5 / tau[normal_case] - d1 * d2 / (2 * tau[normal_case]))
            speed[normal_case] = -gamma[normal_case] / spots[normal_case] * (1 + d1 / (vols[normal_case] * sqr_tau))
            zomma[normal_case] = gamma[normal_case] * ((d1 * d2 - 1) / vols[normal_case])
            ultima[normal_case] = -vega[normal_case] / (vols[normal_case]**2) * (
                d1 * d2 * (1 - d1 * d2) + d1**2 + d2**2)

    if greek_type == 0:
        pv = price
    else:
        pv[:, 0] = price
        pv[:, 1] = delta
        pv[:, 2] = gamma
        pv[:, 3] = theta
        pv[:, 4] = vega
        pv[:, 5] = rho
        pv[:, 6] = vanna
        if greek_type > 1:
            pv[:, 7] = charm
            pv[:, 8] = vomma
            pv[:, 9] = veta
            pv[:, 10] = speed
            pv[:, 11] = zomma
            pv[:, 12] = ultima
    return pv

def inverse_bsm_local(spots, vols, k, tau, r, is_call, ccr, greek_type=1):
    """Calculate BSM greeks in local currency using FX put-call parity, vectorized.
    
    Uses transformations based on FX put-call parity to compute inverse option greeks
    by calling general_bsm with inverted spot/strike and adjusted parameters.
    
    Args:
        spots: Spot price(s) (scalar or array).
        vols: Volatility(s) (scalar or array).
        k: Strike price(s) (scalar or array).
        tau: Time to expiration(s) (scalar or array).
        r: Risk-free rate(s) (scalar or array).
        is_call: Boolean(s) indicating call (True) or put (False) (scalar or array).
        ccr: Continuous carry rate(s) (scalar or array).
        greek_type: 0 for price only, 1 for price and 7 greeks (default=1).
    
    Returns:
        np.ndarray: Price (if greek_type=0) or array of [price, delta, gamma, theta, vega, rho, vanna] (if greek_type=1).
    """
    spots, vols, k, tau, r, ccr, is_call = broadcast_inputs(spots, vols, k, tau, r, ccr, is_call)
    n = len(spots)  # Derive n from broadcasted array length
    
    # Use FX put-call parity: transform inputs and call general_bsm
    this_pv = general_bsm(1 / spots, vols, 1 / k, tau, r - ccr, ~is_call, -ccr, greek_type=min(greek_type, 1))

    if greek_type == 0:
        p_s0 = this_pv * k
        return p_s0
    else:
        # Transform greeks according to VBA logic
        p_s0 = this_pv[:, 0] * k
        delta = -this_pv[:, 1] * k / (spots ** 2)
        gamma = (this_pv[:, 2]  / spots + 2 * this_pv[:, 1]) * k / (spots ** 3)
        theta = this_pv[:, 3] * k
        vega = this_pv[:, 4] * k
        rho = -this_pv[:, 5] * k
        vanna = -this_pv[:, 6] * k / (spots ** 2)
        
        pv = np.zeros((n, 7))
        pv[:, 0] = p_s0
        pv[:, 1] = delta
        pv[:, 2] = gamma
        pv[:, 3] = theta
        pv[:, 4] = vega
        pv[:, 5] = rho
        pv[:, 6] = vanna
        return pv

def general_bsm_iv(prices, spots, k, tau, r, is_call, ccr, n=10, m=10, max_vol=3, cap_vol_at_max=False):
    """Calculate implied volatility using a grid-based iterative approach, fully vectorized.
    
    Divides the volatility range [0, max_vol] into n slices, evaluates option prices
    using general_bsm, and iteratively refines the volatility range over m iterations.
    
    Args:
        prices: Target option price(s) (scalar or array of length k).
        spots: Spot price(s) (scalar or array of length k).
        k: Strike price(s) (scalar or array of length k).
        tau: Time to expiration(s) (scalar or array of length k).
        r: Risk-free rate(s) (scalar or array of length k).
        is_call: Boolean(s) indicating call (True) or put (False) (scalar or array of length k).
        ccr: Continuous carry rate(s) (scalar or array of length k).
        n: Number of volatility slices per iteration (positive integer, default=10).
        m: Number of refinement iterations (positive integer, default=10).
        max_vol: Maximum volatility positive, default=300%)
        cap_vol_at_max: If True, set implied vol to max_vol (3.0) when price > h_pv (default=False).
    
    Returns:
        np.ndarray: Implied volatilities for k options. Returns -1 for invalid cases unless capped.
    
    Raises:
        ValueError: If n or m are not positive integers.
    """
    # Validate n and m as positive scalar integers
    if not (isinstance(n, int) and n > 0):
        raise ValueError("n must be a positive integer")
    if not (isinstance(m, int) and m > 0):
        raise ValueError("m must be a positive integer")

    # Broadcast option-related inputs
    prices, spots, k, tau, r, ccr, is_call = broadcast_inputs(prices, spots, k, tau, r, ccr, is_call)
    k_options = len(spots)  # Number of options
    min_vol = 0.0001  # Minimum volatility (0.01%)
    imp_vols = np.full(k_options, -1.0)  # Initialize output
    active = np.ones(k_options, dtype=bool)  # Track options still being processed
    invalid = np.zeros(k_options, dtype=bool)  # Track definitively invalid options

    # Check initial bounds
    l_pv = general_bsm(spots, np.full(k_options, min_vol), k, tau, r, is_call, ccr, 0)
    h_pv = general_bsm(spots, np.full(k_options, max_vol), k, tau, r, is_call, ccr, 0)
    
    # Handle invalid cases and capping
    below_min = prices < l_pv - 1e-6  # Below minimum price
    above_max = prices > h_pv + 1e-6  # Above maximum price
    invalid |= below_min  # Only prices below l_pv are definitively invalid
    if cap_vol_at_max:
        imp_vols[above_max] = max_vol  # Cap volatility at max_vol
        active[above_max] = False  # No further processing for capped options
    else:
        invalid |= above_max  # Mark as invalid if not capping
    active[invalid] = False

    # Initialize volatility bounds
    vol_bounds = np.zeros((k_options, 2))  # [min_vol, max_vol]
    vol_bounds[:, 0] = min_vol
    vol_bounds[:, 1] = max_vol

    for iteration in range(m):
        if not np.any(active):
            break

        # Create volatility grid
        vol_grid = np.linspace(vol_bounds[:, 0], vol_bounds[:, 1], n + 1).T  # Shape: (k_options, n+1)
        
        # Expand inputs
        spots_exp = np.repeat(spots[:, np.newaxis], n + 1, axis=1)
        k_exp = np.repeat(k[:, np.newaxis], n + 1, axis=1)
        tau_exp = np.repeat(tau[:, np.newaxis], n + 1, axis=1)
        r_exp = np.repeat(r[:, np.newaxis], n + 1, axis=1)
        is_call_exp = np.repeat(is_call[:, np.newaxis], n + 1, axis=1)
        ccr_exp = np.repeat(ccr[:, np.newaxis], n + 1, axis=1)
        
        # Compute option prices
        prices_grid = general_bsm(
            spots_exp.ravel(), vol_grid.ravel(), k_exp.ravel(), tau_exp.ravel(),
            r_exp.ravel(), is_call_exp.ravel(), ccr_exp.ravel(), 0
        ).reshape(k_options, n + 1)

        # Compute price differences
        price_diffs = prices_grid - prices[:, np.newaxis]

        # Identify brackets with tolerance
        above_mask = price_diffs >= -1e-6
        below_mask = price_diffs <= 1e-6
        has_above = np.any(above_mask, axis=1)
        has_below = np.any(below_mask, axis=1)
        bracket_valid = active & has_above & has_below

        # Initialize volatility selections
        below_vol = np.full(k_options, min_vol)
        above_vol = np.full(k_options, max_vol)

        # Find last below and first above indices
        below_indices = np.where(below_mask, np.arange(n + 1)[np.newaxis, :], -1)
        below_max_idx = np.max(below_indices, axis=1)
        valid_below = (below_max_idx >= 0) & bracket_valid
        below_vol[valid_below] = vol_grid[valid_below, below_max_idx[valid_below]]

        above_indices = np.where(above_mask, np.arange(n + 1)[np.newaxis, :], n + 1)
        above_min_idx = np.min(above_indices, axis=1)
        valid_above = (above_min_idx <= n) & bracket_valid
        above_vol[valid_above] = vol_grid[valid_above, above_min_idx[valid_above]]

        # Update volatility bounds
        vol_bounds[:, 0] = np.where(bracket_valid, np.maximum(min_vol, below_vol), vol_bounds[:, 0])
        vol_bounds[:, 1] = np.where(bracket_valid, np.minimum(max_vol, above_vol), vol_bounds[:, 1])

        # Check for convergence
        converged = (vol_bounds[:, 1] - vol_bounds[:, 0] < 1e-6) & bracket_valid
        imp_vols[converged] = (vol_bounds[converged, 0] + vol_bounds[converged, 1]) / 2
        active[converged] = False

        # Only mark as invalid after repeated bracketing failures (at final iteration)
        if iteration == m - 1:
            active[~bracket_valid] = False

    # Set final implied volatilities for remaining active options
    imp_vols[active] = (vol_bounds[active, 0] + vol_bounds[active, 1]) / 2

    # Final edge case handling: only invalid options get -1
    imp_vols[invalid] = -1
    imp_vols[imp_vols <= min_vol] = -1
    if not cap_vol_at_max:
        imp_vols[imp_vols >= max_vol] = -1

    return imp_vols