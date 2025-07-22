import pandas as pd
import numpy as np

# Load the datasets
def load_data():
    fraud_df = pd.read_csv('../data/raw/Fraud_Data.csv')
    credit_df = pd.read_csv("../data/raw/creditcard.csv")
    ip_df = pd.read_csv("../data/raw/IpAddress_to_Country.csv")
    return fraud_df, credit_df, ip_df

# Data Cleaning
def clean_data(fraud_df, credit_df, ip_df):
    # Remove duplicates
    fraud_df = fraud_df.drop_duplicates()
    credit_df = credit_df.drop_duplicates()
    ip_df = ip_df.drop_duplicates()
    
    # Handle missing values (example: forward fill)
    fraud_df = fraud_df.fillna(method='ffill')
    credit_df = credit_df.fillna(method='ffill')
    ip_df = ip_df.fillna(method='ffill')
    
    return fraud_df, credit_df, ip_df

# Feature Engineering
def feature_engineering(fraud_df, credit_df):
    # Example: Create a new feature in fraud_df
    fraud_df['new_feature'] = fraud_df['existing_feature'] * 2  # Modify as needed
    
    # Example: Merge with IP data if relevant
    # fraud_df = fraud_df.merge(ip_df, on='ip_address', how='left')  # Example merge
    
    return fraud_df, credit_df

# Data Analysis
def analyze_data(fraud_df, credit_df):
    # Descriptive statistics
    print("Fraud Data Statistics:")
    print(fraud_df.describe())
    
    print("\nCredit Card Data Statistics:")
    print(credit_df.describe())
    
    # Example: Correlation matrix
    correlation_matrix = credit_df.corr()
    print("\nCredit Card Correlation Matrix:")
    print(correlation_matrix)

# Main function
if __name__ == "__main__":
    fraud_df, credit_df, ip_df = load_data()
    fraud_df, credit_df, ip_df = clean_data(fraud_df, credit_df, ip_df)
    fraud_df, credit_df = feature_engineering(fraud_df, credit_df)
    analyze_data(fraud_df, credit_df)