import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1. GENERATE SYNTHETIC DATASET
# ==========================================
def generate_insurance_data(n_samples=10000):
    np.random.seed(42)[ 5 (https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGVbm1JUw6IUOiINtnx-yCDChd004Gh7_VFvg-ysAyWPNqTKZILHObNQLqLZ-5L5JY2Y4tc0TRtY3nCRZGKuHWEZlAa1g_Ph8YtfLcxCQF-282PdE8zR7KKL-BLmFNFAl2KhR2tsp8=)]
    
    data = {
        'customer_id': np.arange(1000, 1000 + n_samples),
        'age': np.random.randint(18, 80, n_samples),
        'annual_income': np.random.normal(55000, 15000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'months_as_customer': np.random.randint(1, 120, n_samples),
        'policy_premium': np.random.normal(1200, 300, n_samples),
        'number_of_past_claims': np.random.randint(0, 5, n_samples),
        'payment_method': np.random.choice(['Auto-Pay', 'Credit Card', 'Check', 'Cash'], n_samples),
        'marital_status': np.random.choice(['Married', 'Single', 'Divorced'], n_samples)
    }[ 7 (https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHaJHQ9noYdEdESgnMZPbEX-KTBdPpqH1KPpCtEDUgjXUCR7pYVPkx_mgp9Rd2LzPe8yhqWVBK-XVtw0lbIkAm8spobObaaSCkkaFZl8KC6dwyt-bNmm34MR8xSepAkrL9Stqm6iKbXAbhTV9UBVB7NQPTfco4hv6dXIPjtsbrTFA==)][ 9 (https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHsvKb7DbXP_jlcTypgJ44EX5VMOUfDOGAwCB2GiBBKS-SRVxtq-qu2KtbuP4fklbL_rnpjRpnqIao401-_hWjCzlDY88Ect8qvau3WyBU6A5mUGTTF-1OsU7W_LXrTV-oFtZdRThxfsG02Y77uCOhyzznEDRyN_e3D)]
    
    df = pd.DataFrame(data)[ 2 (https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFMMGKPEPZ7qcG9BwfibivTGu9DOeOgDACqj6q-XARViuvEZ5dPityPzsf3DiyiBQbt8B1GjH8fQ3Tqirz4O_JPJrXYGS4ZCWaUKWbz5ag3bST3tKmWhlMrur9MJAEz965rqjfDob31vYl35sIK6otx4WPxS6DLA_QwBgDXGDIqQvOxMG_ubYMDJu_6)][ 3 (https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFOCCswAbH88WEmjD2qHE_1rbPGmsiG9Tm6z4IaParXcWSKh2nB5jNcnyhhz31_ITOsfAZiVZASseP6ZrwHwfA2-hH_DkhQAn_soHg4tvRQLLyq_sRRXEq0UMUFe8HkvyxbYQmaTUg=)][ 5 (https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGVbm1JUw6IUOiINtnx-yCDChd004Gh7_VFvg-ysAyWPNqTKZILHObNQLqLZ-5L5JY2Y4tc0TRtY3nCRZGKuHWEZlAa1g_Ph8YtfLcxCQF-282PdE8zR7KKL-BLmFNFAl2KhR2tsp8=)]
    
    # Introduce logic for 'default' target variable to make the model learnable
    # People with low credit scores, low income, and high premiums are more likely to default
    df['default_probability'] = 0.1  # Base risk
    
    # Risk factors
    df.loc[df['credit_score'] < 580, 'default_probability'] += 0.4
    df.loc[df['credit_score'] < 670, 'default_probability'] += 0.2
    df.loc[df['annual_income'] < 30000, 'default_probability'] += 0.2
    df.loc[df['payment_method'] == 'Cash', 'default_probability'] += 0.15
    df.loc[df['policy_premium'] > 1800, 'default_probability'] += 0.1
    
    # Cap probability at 1.0
    df['default_probability'] = df['default_probability'].clip(upper=1.0)[ 5 (https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGVbm1JUw6IUOiINtnx-yCDChd004Gh7_VFvg-ysAyWPNqTKZILHObNQLqLZ-5L5JY2Y4tc0TRtY3nCRZGKuHWEZlAa1g_Ph8YtfLcxCQF-282PdE8zR7KKL-BLmFNFAl2KhR2tsp8=)]
    
    # Generate binary target based on probability
    df['defaulted'] = np.random.binomial(1, df['default_probability'])
    
    # Drop the probability column (we won't have this in real life)
    df = df.drop(columns=['default_probability', 'customer_id'])
    
    return df[ 5 (https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGVbm1JUw6IUOiINtnx-yCDChd004Gh7_VFvg-ysAyWPNqTKZILHObNQLqLZ-5L5JY2Y4tc0TRtY3nCRZGKuHWEZlAa1g_Ph8YtfLcxCQF-282PdE8zR7KKL-BLmFNFAl2KhR2tsp8=)]

# Create the dataset
print("Generating synthetic data...")
df = generate_insurance_data()
print(f"Dataset Shape: {df.shape}")
print(df.head())[ 5 (https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGVbm1JUw6IUOiINtnx-yCDChd004Gh7_VFvg-ysAyWPNqTKZILHObNQLqLZ-5L5JY2Y4tc0TRtY3nCRZGKuHWEZlAa1g_Ph8YtfLcxCQF-282PdE8zR7KKL-BLmFNFAl2KhR2tsp8=)]

# ==========================================
# 2. DATA PREPROCESSING[ 4 (https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFwpZARZDs2uq5oqbLODksoWBQ2WTH1YbtmaIk5uonHYWpEKSTcZDWzlzvHdXIGGxipMhj9_8re-UjwKN2xyg9bh90vOBFcszoN3kfH_xulcofE9t2s4VUcGiKnKCMauA6Kj401QIA5VgtplctCzA==)][ 5 (https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGVbm1JUw6IUOiINtnx-yCDChd004Gh7_VFvg-ysAyWPNqTKZILHObNQLqLZ-5L5JY2Y4tc0TRtY3nCRZGKuHWEZlAa1g_Ph8YtfLcxCQF-282PdE8zR7KKL-BLmFNFAl2KhR2tsp8=)][ 6 (https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHWOlVy9DgFh2hQzPDq-c5gFLi86oxSpfDTIntLOB0auXfu5EqiDL-s7-E6qoP4E7fSwNzCllD7FfaLJjnUoSUuBwF1Gh_DV1i-7dUSVtim1KscUznAKRH36Yr8Uw72Rq_XMyqYMruWTs7CtNuaWEx2uguRHCLPFxxt8Em0-e48mR-nhwGxvmgDLjdMHlefd5qYetTUAPh3Qa8BHwp4Aw==)]
# ==========================================

# Separate Features (X) and Target (y)
X = df.drop('defaulted', axis=1)
y = df['defaulted'][ 5 (https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGVbm1JUw6IUOiINtnx-yCDChd004Gh7_VFvg-ysAyWPNqTKZILHObNQLqLZ-5L5JY2Y4tc0TRtY3nCRZGKuHWEZlAa1g_Ph8YtfLcxCQF-282PdE8zR7KKL-BLmFNFAl2KhR2tsp8=)]

# Encode Categorical Variables (One-Hot Encoding)
X = pd.get_dummies(X, drop_first=True)

# Split into Training and Testing Sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Numerical Features
# (Tree-based models like Random Forest don't strictly require scaling, 
# but it's good practice for interpretability and if we switch models later)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)[ 5 (https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGVbm1JUw6IUOiINtnx-yCDChd004Gh7_VFvg-ysAyWPNqTKZILHObNQLqLZ-5L5JY2Y4tc0TRtY3nCRZGKuHWEZlAa1g_Ph8YtfLcxCQF-282PdE8zR7KKL-BLmFNFAl2KhR2tsp8=)]

# ==========================================
# 3. TRAIN THE MODEL (Random Forest)
# ==========================================
print("\nTraining Random Forest Model...")
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

# ==========================================
# 4. EVALUATE PERFORMANCE
# ==========================================
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance Visualization
feature_names = X.columns
importances = model.feature_importances_
indices = np.argsort(importances)[::-1][ 5 (https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGVbm1JUw6IUOiINtnx-yCDChd004Gh7_VFvg-ysAyWPNqTKZILHObNQLqLZ-5L5JY2Y4tc0TRtY3nCRZGKuHWEZlAa1g_Ph8YtfLcxCQF-282PdE8zR7KKL-BLmFNFAl2KhR2tsp8=)]

plt.figure(figsize=(10, 6))
plt.title("Feature Importance: What drives defaults?")
plt.bar(range(X.shape 1 ), importances[indices], align="center")
plt.xticks(range(X.shape 1 ), feature_names[indices], rotation=90)
plt.tight_layout()
plt.show()

# ==========================================
# 5. PREDICTION SYSTEM FOR NEW CUSTOMERS
# ==========================================
def predict_default_risk(new_customer_data):
    """
    Takes a dictionary of customer details and returns default probability.[ 1 (https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEoVYSPP1QV4QmnAJWOGnKL1BR4hTfSGgtuRmBPzc1LpaRvud-e5wN26NKIBIgrIvFV11EQOLhQCtxJ3od5dKbts-Cy3G9GITp4ZKZpJHUunyGD857wKW8BgExw0oO3s5scI8VUiLvPukgtezYBawjICWCktAlJT2hO-O6-DbDNai7qdfSyv0JlmCQeA7Q5Kk-sPIdqpg==)][ 5 (https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGVbm1JUw6IUOiINtnx-yCDChd004Gh7_VFvg-ysAyWPNqTKZILHObNQLqLZ-5L5JY2Y4tc0TRtY3nCRZGKuHWEZlAa1g_Ph8YtfLcxCQF-282PdE8zR7KKL-BLmFNFAl2KhR2tsp8=)][ 8 (https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHfzLxqVvTiRPa7zhGzHDUTm4NRpSIuCc36tkvx-5oARWiF5X1p9Tmi3O6P3-HRya6qpA8gbsGxnm2-k6k07wzzl1uLwMtd1UiZjnhHVe_iodIYofG4lYV1hhw9eg7TrwdO9mpWpgT8ajj-NsjUjFO9)]
    """
    # Convert dict to DataFrame
    input_df = pd.DataFrame([new_customer_data])[ 5 (https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGVbm1JUw6IUOiINtnx-yCDChd004Gh7_VFvg-ysAyWPNqTKZILHObNQLqLZ-5L5JY2Y4tc0TRtY3nCRZGKuHWEZlAa1g_Ph8YtfLcxCQF-282PdE8zR7KKL-BLmFNFAl2KhR2tsp8=)]
    
    # One-Hot Encode (Align with training columns)
    input_df = pd.get_dummies(input_df)[ 5 (https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGVbm1JUw6IUOiINtnx-yCDChd004Gh7_VFvg-ysAyWPNqTKZILHObNQLqLZ-5L5JY2Y4tc0TRtY3nCRZGKuHWEZlAa1g_Ph8YtfLcxCQF-282PdE8zR7KKL-BLmFNFAl2KhR2tsp8=)]
    
    # Ensure all columns from training exist in input (fill missing with 0)
    input_df = input_df.reindex(columns=X.columns, fill_value=0)
    
    # Scale features
    input_scaled = scaler.transform(input_df)[ 5 (https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGVbm1JUw6IUOiINtnx-yCDChd004Gh7_VFvg-ysAyWPNqTKZILHObNQLqLZ-5L5JY2Y4tc0TRtY3nCRZGKuHWEZlAa1g_Ph8YtfLcxCQF-282PdE8zR7KKL-BLmFNFAl2KhR2tsp8=)]
    
    # Predict
    prediction = model.predict(input_scaled) 0 
    probability = model.predict_proba(input_scaled) 0  1 [ 5 (https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGVbm1JUw6IUOiINtnx-yCDChd004Gh7_VFvg-ysAyWPNqTKZILHObNQLqLZ-5L5JY2Y4tc0TRtY3nCRZGKuHWEZlAa1g_Ph8YtfLcxCQF-282PdE8zR7KKL-BLmFNFAl2KhR2tsp8=)]
    
    return prediction, probability[ 10 (https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHwQSg8LSIqsdeF2lmmsdKtKn2F4DjwBDvrjhEXnaj-uxG9JhWzlZjY4H38irCshTkYmR1_Uivf_nmcHUgVgT6RHkb0Y5xH-5NxYepmyfH4yZxten2KouK4_dxAz4tKtz9RhfxpygaMZs1_aJFxDkRUVO95euZHS0bOGWVeSAQrbGQzgnvxXLJFN7Z5HayF_wsZ6ODinc1W0SNTGFdx3tAMt-UUDoMGdewiuQjVP4UxEfqe-KdWuk2VHA==)]

# Example Usage
new_person = {
    'age': 35,
    'annual_income': 28000,      # Low income (Risk factor)
    'credit_score': 550,         # Low credit score (High Risk)
    'months_as_customer': 12,
    'policy_premium': 1500,
    'number_of_past_claims': 2,
    'payment_method': 'Cash',    # Cash payments often correlate with higher risk
    'marital_status': 'Single'
}[ 5 (https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGVbm1JUw6IUOiINtnx-yCDChd004Gh7_VFvg-ysAyWPNqTKZILHObNQLqLZ-5L5JY2Y4tc0TRtY3nCRZGKuHWEZlAa1g_Ph8YtfLcxCQF-282PdE8zR7KKL-BLmFNFAl2KhR2tsp8=)]

pred, prob = predict_default_risk(new_person)

print("\n--- New Customer Prediction ---")
print(f"Customer Profile: {new_person}")
print(f"Likelihood of Default: {prob*100:.2f}%")
print(f"Model Recommendation: {'HIGH RISK - DO NOT INSURE' if pred == 1 else 'SAFE TO INSURE'}")