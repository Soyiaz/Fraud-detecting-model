# Fraud Detection Project

This project is dedicated to advancing fraud detection for both e-commerce and credit card transactions. It explores a variety of machine learning approaches, addresses class imbalance, and incorporates model explainability tools.

## Datasets Utilized
- Fraud_Data.csv
- IpAddress_to_Country.csv
- creditcard.csv

## 🧠 Data Analysis & Preparation

The data preparation process involved cleaning, feature engineering, and merging information from three datasets:

- `Fraud_Data.csv`: Contains user, transaction, and fraud label data.
- `IpAddress_to_Country.csv`: Associates IP addresses with countries.
- `creditcard.csv`: Offers anonymized credit card transaction records for additional fraud analysis.

### ✅ Completed Workflow

#### 1. Addressing Missing Data
- Reviewed all datasets for missing entries.
- Removed or imputed missing values as appropriate.

#### 2. Data Cleaning
- Deleted duplicate records.
- Ensured correct data types (e.g., parsed `signup_time` and `purchase_time` as datetime).
- Converted IP addresses to integer format for dataset merging.

#### 3. Exploratory Data Analysis (EDA)
- Conducted univariate analysis (e.g., class distribution, feature histograms).
- Performed bivariate analysis (e.g., fraud rates by country and by time of day).

#### 4. Merging for Geolocation
- Transformed IP ranges in `IpAddress_to_Country.csv` to numeric values.
- Merged `Fraud_Data.csv` with country data based on IP ranges to enrich with location features.

#### 5. Feature Engineering
- Extracted hour and weekday from `purchase_time`.
- Calculated `time_since_signup` (interval between signup and purchase).
- Developed features for transaction frequency and velocity to capture user behavior patterns.

#### 6. Data Transformation

To prepare for modeling, the following steps were executed:

- **Train-Test Split**: Divided the data into 80% training and 20% testing sets, using stratified sampling to maintain class balance.

- **Column Identification**:
  - Categorical columns: Identified by type (`object`, `category`).
  - Numerical columns: Identified by type (`int64`, `float64`).

- **Preprocessing Pipelines**:
  - **Numerical** features were standardized with `StandardScaler`.
  - **Categorical** features were one-hot encoded using `OneHotEncoder` with `handle_unknown='ignore'`.

- **Class Imbalance Handling**:
  - Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to the training set after preprocessing, generating synthetic samples for the minority fraud class to balance the data.

- **Final Outputs**:
  - `X_train_resampled`, `y_train_resampled`: Preprocessed and balanced training data.
  - `X_test_processed`, `y_test`: Preprocessed test data.