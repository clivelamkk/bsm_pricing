import pandas as pd

def transform_laevitas_df(df):
    """
    Transforms a DataFrame from Laevitas API format to a format compatible with gen_CurveIV.
    
    Args:
        df (pd.DataFrame): Input DataFrame with Laevitas API columns including 'date', 
                           'expiration_date', 'underlyer', 'exercise', 'settlement', 
                           'strike', 'forward_price', 'claim_type', and value columns.
    
    Returns:
        pd.DataFrame: Transformed DataFrame with columns renamed and pivoted for gen_CurveIV,
                      including 'underlying', 'date', 'expiry', 'strike', 'spot', 
                      'call_bid', 'put_bid', 'call_ask', 'put_ask', etc.
    """
    # Create a copy to avoid modifying the input DataFrame
    df = df.copy()

    # Convert date columns to datetime format
    df['expiration_date_dt'] = pd.to_datetime(df['expiration_date'], utc=True)
    df['date_dt'] = pd.to_datetime(df['date'].astype(float), unit='ms', utc=True)
    
    # Calculate tenor in years from date and expiry
    df['tenor'] = (df['expiration_date_dt'] - df['date_dt']).dt.total_seconds() / (365.0 * 24 * 60 * 60)
    
    # Define columns to use as index for pivoting
    index_cols = ['underlyer', 'exercise', 'settlement', 
                  'strike', 'forward_price', 'expiration_date_dt', 
                  'date_dt']
    
    # Define columns to pivot (values that differ between call and put)
    value_cols = ['best_bid_price', 'best_ask_price', 'volume', 'open_interest', 
                  'best_bid_amount', 'best_ask_amount']

    # Pivot the DataFrame to separate call and put data into columns
    pivoted_df = df.pivot(index=index_cols, 
                          columns='claim_type', 
                          values=value_cols).reset_index()

    # Flatten multi-level column names after pivoting
    pivoted_df.columns = ['_'.join(col).strip() if col[1] else col[0] 
                          for col in pivoted_df.columns.values]

    # Rename columns to match gen_CurveIV expected format
    rename_dict = {
        'underlyer': 'underlying',
        'forward_price': 'spot',
        'date_dt': 'date',
        'expiration_date_dt': 'expiry',
        'claim_type': 'cp_type',
        'best_bid_price_call': 'call_bid',
        'best_bid_price_put': 'put_bid',
        'best_ask_price_call': 'call_ask',
        'best_ask_price_put': 'put_ask',
        'volume_call': 'call_volume',
        'volume_put': 'put_volume',
        'open_interest_call': 'call_oi',
        'open_interest_put': 'put_oi',
        'best_bid_amount_call': 'call_bid_amount',
        'best_bid_amount_put': 'put_bid_amount',
        'best_ask_amount_call': 'call_ask_amount',
        'best_ask_amount_put': 'put_ask_amount'
    }
    pivoted_df = pivoted_df.rename(columns=rename_dict)

    return pivoted_df
