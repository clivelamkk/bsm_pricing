import pandas as pd
import numpy as np
from bsm_pricing import *


# ------------------------------------------------
def gen_CurveIV(df, isInverseQuoted):   
    """
    Generates implied volatility curves from options data, calculating implied forwards, interest rates,
    and volatilities for calls and puts.
    
    Args:
        df (pd.DataFrame): Input DataFrame with columns: date, underlying, expiry, spot, strike, tenor,
                           call_bid, call_ask, put_bid, put_ask.
        isInverseQuoted (bool): True if prices are quoted in underlying asset units, False if in USD.
    
    Returns:
        pd.DataFrame: DataFrame with implied volatility curves, including bid/ask volatilities,
                      variances, and moneyness.
    """
    df = df.copy()

    # Calculate tenor in years from date and expiry
    df['tenor'] = (df['expiry'] - df['date']).dt.total_seconds() / (365.0 * 24 * 60 * 60)

    # Calculate implied forward prices based on put-call parity
    if isInverseQuoted:
        df['implied_forward_ask'] = np.where(
            (df[['call_ask', 'put_bid', 'strike']].notnull().all(axis=1)) & 
            (df['call_ask'] > 0) & 
            (df['put_bid'] > 0),
            df['strike'] / (1 - df['call_ask'] + df['put_bid']),
            np.nan
        )
        df['implied_forward_bid'] = np.where(
            (df[['call_bid', 'put_ask', 'strike']].notnull().all(axis=1)) & 
            (df['call_bid'] > 0) & 
            (df['put_ask'] > 0),
            df['strike'] / (1 - df['call_bid'] + df['put_ask']),
            np.nan
        )
    else:
        df['implied_forward_ask'] = np.where(
            (df[['call_ask', 'put_bid', 'strike']].notnull().all(axis=1)) & 
            (df['call_ask'] > 0) & 
            (df['put_bid'] > 0),
            df['strike'] + (df['call_ask'] - df['put_bid']),
            np.nan
        )
        df['implied_forward_bid'] = np.where(
            (df[['call_bid', 'put_ask', 'strike']].notnull().all(axis=1)) & 
            (df['call_bid'] > 0) & 
            (df['put_ask'] > 0),
            df['strike'] + (df['call_bid'] - df['put_ask']),
            np.nan
        )

    # Group by date, underlying, and expiry to compute averages
    grouped_df = df.groupby(['date', 'underlying', 'expiry']).agg({
        'spot': 'mean',
        'tenor': 'mean',
        'implied_forward_bid': 'max',
        'implied_forward_ask': 'min'
    }).reset_index()
    grouped_df['imp_fwd'] = (grouped_df['implied_forward_bid'] + grouped_df['implied_forward_ask']) / 2

    # Calculate implied interest rate: ln(fwd/spot)/tenor
    valid_mask = (
        (grouped_df['imp_fwd'] > 0) &
        (grouped_df['spot'] > 0) &
        (grouped_df['tenor'] != 0)
    )
    grouped_df['int_rate'] = np.nan
    grouped_df.loc[valid_mask, 'int_rate'] = (
        np.log(grouped_df.loc[valid_mask, 'imp_fwd'] / 
               grouped_df.loc[valid_mask, 'spot']) / 
        grouped_df.loc[valid_mask, 'tenor']
    )

    # Drop temporary forward columns
    grouped_df.drop(['implied_forward_bid', 'implied_forward_ask'], axis=1, inplace=True)

    # Merge interest rate and implied forward back into main DataFrame
    df = df.merge(
        grouped_df[['date', 'underlying', 'expiry', 'imp_fwd', 'int_rate']],
        on=['date', 'underlying', 'expiry'],
        how='left'
    )

    # Convert prices for inverse-quoted assets if needed
    spots = np.array(df['spot'])
    if isInverseQuoted:
        call_bid_prices = np.array(df['call_bid']) * spots
        call_ask_prices = np.array(df['call_ask']) * spots
        put_bid_prices = np.array(df['put_bid']) * spots
        put_ask_prices = np.array(df['put_ask']) * spots
    else:
        call_bid_prices = np.array(df['call_bid'])
        call_ask_prices = np.array(df['call_ask'])
        put_bid_prices = np.array(df['put_bid'])
        put_ask_prices = np.array(df['put_ask'])

    # Calculate implied volatilities using Black-Scholes-Merton model
    k = np.array(df['strike'])
    tau = np.array(df['tenor'])
    r = np.array(df['int_rate'])
    r[pd.isna(r)] = 0
    ccr = r

    call_bid_vol = general_bsm_iv(call_bid_prices, spots, k, tau, r, True, ccr)
    call_ask_vol = general_bsm_iv(call_ask_prices, spots, k, tau, r, True, ccr)
    put_bid_vol = general_bsm_iv(put_bid_prices, spots, k, tau, r, False, ccr)
    put_ask_vol = general_bsm_iv(put_ask_prices, spots, k, tau, r, False, ccr)

    df['call_bid_vol'] = call_bid_vol
    df['call_ask_vol'] = call_ask_vol
    df['put_bid_vol'] = put_bid_vol
    df['put_ask_vol'] = put_ask_vol

    # Create volatility curve DataFrame
    curve_df = df.copy()

    # Handle invalid volatilities
    curve_df.loc[curve_df['call_ask_vol'] < 0, 'call_ask_vol'] = 100000
    curve_df.loc[curve_df['put_ask_vol'] < 0, 'put_ask_vol'] = 100000

    # Compute combined bid and ask volatilities
    curve_df['bid_vol'] = np.maximum(curve_df['call_bid_vol'], curve_df['put_bid_vol'])
    curve_df['ask_vol'] = np.minimum(curve_df['call_ask_vol'], curve_df['put_ask_vol'])

    # Drop unnecessary columns
    dropCol = ['call_bid', 'put_bid', 'call_ask', 'put_ask', 
               'call_bid_vol', 'put_bid_vol', 'call_ask_vol', 'put_ask_vol',
               'implied_forward_ask', 'implied_forward_bid']
    curve_df.drop(dropCol, axis=1, inplace=True)

    # Replace extreme ask volatilities with NaN
    curve_df.loc[curve_df['ask_vol'] >= 100000, 'ask_vol'] = -1
    curve_df['bid_vol'] = curve_df['bid_vol'].replace(-1, np.nan)
    curve_df['ask_vol'] = curve_df['ask_vol'].replace(-1, np.nan)
    
    # Calculate variance and moneyness
    curve_df['bid_var'] = curve_df['tenor'] * (curve_df['bid_vol'] ** 2)
    curve_df['ask_var'] = curve_df['tenor'] * (curve_df['ask_vol'] ** 2)
    curve_df['moneyness'] = curve_df['strike'] / curve_df['spot']

    # Sort by key columns for consistency
    curve_df = curve_df.sort_values(by=['underlying', 'date', 'expiry', 'strike'])

    return curve_df
