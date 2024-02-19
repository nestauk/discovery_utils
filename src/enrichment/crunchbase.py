"""
discovery_utils.src.getters.crunchbase.py

Module for data enrichment utils for CB data.

"""        

# Define global variables 
import pandas as pd


# EMPLOYEE SIZE
# to be used with organizations.csv

def _employee_count_threshold(employee_count: int, threshold: int) -> bool:
    """Check if the employee count is below a certain threshold and return a boolean."""
    return employee_count < threshold


# GEOGRAPHICAL LOCATION
# to be used with organizations.csv

def _country(actual_code: str, expected_code: str) -> bool:
    """Check if the country code is the expected one and return a boolean."""
    return actual_code == expected_code
    

# TOTAL FUNDING GBP CONVERSION
# to be used with organizations.csv



def total_funding_gbp_conversion(data, total_funding_col='total_funding_usd'):
    """Convert total funding from USD to GBP"""
    if total_funding_col in data.columns:
        data['total_funding_gbp'] = data[total_funding_col] * 0.72
    else:
        print(f"Warning: '{total_funding_col}' column not found in DataFrame.")


# TOTAL FUNDING SIZE
# to be used with an already created "total_funding_gbp" column in organizations.csv

# Building block function - not to be used directly

def categorize_total_funding(data, total_funding_col='total_funding_gbp'):
    """Categorise total funding based on the 'total_funding_gbp' column"""
    if total_funding_col in data.columns:
        data['total_funding_size'] = pd.cut(data[total_funding_col],
                                            bins=[0, 1000000, 5000000, float('inf')],
                                            labels=['potential_investment_opp', 'investment_opp', 'too_big_to_invest'])
    else:
        print(f"Warning: '{total_funding_col}' column not found in DataFrame.")

# Master functions - to be used directly

def total_funding_size_table(df: pd.DataFrame) -> None:
    """Categorise total funding for the whole DataFrame"""
    categorize_total_funding(df)

def total_funding_size_record(record: pd.Series) -> None:
    """Categorise total funding for a specific record"""
    categorize_total_funding(record)