"""
Kaggle House Prices Competition - Full Pipeline Test

This script:
1. Registers the Kaggle training data
2. Invokes the training agent with no model preference (LLM chooses)
3. Stores the resulting model
4. Loads the Kaggle test data, applies transformations, and generates predictions
5. Creates a submission.csv file for Kaggle

Run: python .kaggle/house-prices/run_kaggle_competition.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add paths for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "data-tools"))
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "models-tools" / "training"))

from utils import register_dataset, get_registered_dataset
from agents.training.agent import invoke_training_agent
from model_storage import load_model, get_model_info


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
SUBMISSION_FILE = DATA_DIR / "submission.csv"

# Use existing model instead of training (set to None to train new)
USE_EXISTING_MODEL = None  # e.g., "xgboost_1769321495_v1"

# The goal prompt - crafted to guide the agent for this specific competition
GOAL = """
Predict residential home sale prices (SalePrice) in Ames, Iowa using the 79 housing features provided.

This is a REGRESSION task. The target variable is SalePrice (continuous, in dollars).
The evaluation metric is RMSE between log(predicted) and log(actual), so consider:
- Log-transforming the target variable for training
- Features that capture multiplicative effects on price

Key feature categories to explore:
- Quality ratings (OverallQual, ExterQual, KitchenQual, etc.) - strong predictors
- Size features (GrLivArea, TotalBsmtSF, GarageArea, LotArea)
- Location/Neighborhood effects
- Age and condition (YearBuilt, YearRemodAdd, OverallCond)
- Amenities (Fireplaces, PoolArea, central air)

Feature engineering opportunities:
- Total square footage (basement + above ground)
- Age of house at sale (YrSold - YearBuilt)
- Quality x Size interactions
- Bathroom counts (full + half baths)
- Has garage/pool/fireplace flags

Handle missing values appropriately - many NA values in this dataset mean "not applicable" 
(e.g., NA in PoolQC means no pool, not missing data).

Prioritize models that work well on tabular regression: XGBoost, Random Forest, or GLM.
""".strip()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_header(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_step(step: str, msg: str = ""):
    """Print a step indicator."""
    print(f"\n[{step}] {msg}")


def preview_dataframe(df: pd.DataFrame, name: str, max_rows: int = 3):
    """Print a preview of a DataFrame."""
    print(f"\n  📊 {name}")
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}")
    print(f"  Dtypes summary: {df.dtypes.value_counts().to_dict()}")
    if max_rows > 0:
        print(f"  First {max_rows} rows:")
        print(df.head(max_rows).to_string(max_cols=8))


def apply_feature_engineering(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Apply feature engineering to match what the trained model expects.
    
    The model's feature_names have 'num__' prefix, but the input data should have
    the raw column names (without prefix) since the model's preprocessor handles that.
    """
    df = df.copy()
    
    # Extract the required column names (strip 'num__' and 'cat__' prefixes)
    required_cols = set()
    for f in feature_names:
        if f.startswith('num__'):
            required_cols.add(f[5:])  # Remove 'num__' prefix
        elif f.startswith('cat__'):
            required_cols.add(f[5:])  # Remove 'cat__' prefix
        else:
            required_cols.add(f)
    
    print(f"  Model expects {len(required_cols)} input columns")
    
    # Fill missing values for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
    
    # Fill missing values for categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna('None')
    
    # 1. Log transformations
    for col in ['LotFrontage', 'LotArea', 'MasVnrArea']:
        if col in df.columns:
            df[f'{col}_log1p'] = np.log1p(df[col].fillna(0))
    
    # 2. Binning MSSubClass
    if 'MSSubClass' in df.columns:
        df['MSSubClass_binned'] = pd.cut(
            df['MSSubClass'], 
            bins=[0, 30, 60, 90, 200], 
            labels=[0, 1, 2, 3]
        ).astype(float).fillna(0)
    
    # 3. One-hot encoding for categorical columns
    # MSZoning
    if 'MSZoning' in df.columns:
        for cat in ['FV', 'RH', 'RL', 'RM']:
            df[f'MSZoning_ohe_{cat}'] = (df['MSZoning'] == cat).astype(int)
    
    # Neighborhood
    if 'Neighborhood' in df.columns:
        neighborhoods = ['Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr', 
                        'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV',
                        'Mitchel', 'NAmes', 'NPkVill', 'NWAmes', 'NoRidge',
                        'NridgHt', 'OldTown', 'SWISU', 'Sawyer', 'SawyerW',
                        'Somerst', 'StoneBr', 'Timber', 'Veenker']
        for cat in neighborhoods:
            df[f'Neighborhood_ohe_{cat}'] = (df['Neighborhood'] == cat).astype(int)
    
    # BldgType
    if 'BldgType' in df.columns:
        for cat in ['2fmCon', 'Duplex', 'Twnhs', 'TwnhsE']:
            df[f'BldgType_ohe_{cat}'] = (df['BldgType'] == cat).astype(int)
    
    # HouseStyle
    if 'HouseStyle' in df.columns:
        for cat in ['1.5Unf', '1Story', '2.5Fin', '2.5Unf', '2Story', 'SFoyer', 'SLvl']:
            df[f'HouseStyle_ohe_{cat}'] = (df['HouseStyle'] == cat).astype(int)
    
    # 4. Computed features
    # Total square footage
    basement_sf = df['TotalBsmtSF'].fillna(0) if 'TotalBsmtSF' in df.columns else 0
    first_floor = df['1stFlrSF'].fillna(0) if '1stFlrSF' in df.columns else 0
    second_floor = df['2ndFlrSF'].fillna(0) if '2ndFlrSF' in df.columns else 0
    df['TotalSF'] = basement_sf + first_floor + second_floor
    
    # Total bathrooms
    full_bath = df['FullBath'].fillna(0) if 'FullBath' in df.columns else 0
    half_bath = df['HalfBath'].fillna(0) if 'HalfBath' in df.columns else 0
    bsmt_full = df['BsmtFullBath'].fillna(0) if 'BsmtFullBath' in df.columns else 0
    bsmt_half = df['BsmtHalfBath'].fillna(0) if 'BsmtHalfBath' in df.columns else 0
    df['TotalBath'] = full_bath + 0.5 * half_bath + bsmt_full + 0.5 * bsmt_half
    
    # Binary flags
    df['HasGarage'] = (df['GarageArea'].fillna(0) > 0).astype(int) if 'GarageArea' in df.columns else 0
    df['HasFireplace'] = (df['Fireplaces'].fillna(0) > 0).astype(int) if 'Fireplaces' in df.columns else 0
    df['HasPool'] = (df['PoolArea'].fillna(0) > 0).astype(int) if 'PoolArea' in df.columns else 0
    
    # Interaction features
    if 'OverallQual' in df.columns and 'GrLivArea' in df.columns:
        df['OverallQual_GrLivArea'] = df['OverallQual'] * df['GrLivArea']
    if 'OverallQual' in df.columns:
        df['OverallQual_TotalSF'] = df['OverallQual'] * df['TotalSF']
    if 'GarageCars' in df.columns and 'GrLivArea' in df.columns:
        df['GarageCars_GrLivArea'] = df['GarageCars'].fillna(0) * df['GrLivArea']
    
    # 5. Ordinal encodings for quality columns
    quality_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    yes_no_map = {'N': 0, 'Y': 1}
    exposure_map = {'No': 0, 'Mn': 1, 'Av': 2, 'Gd': 3}
    finish_map = {'Unf': 0, 'RFn': 1, 'Fin': 2}
    
    if 'CentralAir' in df.columns:
        df['CentralAir_ordinal'] = df['CentralAir'].map(yes_no_map).fillna(0)
    if 'HeatingQC' in df.columns:
        df['HeatingQC_ordinal'] = df['HeatingQC'].map(quality_map).fillna(3)
    if 'KitchenQual' in df.columns:
        df['KitchenQual_ordinal'] = df['KitchenQual'].map(quality_map).fillna(3)
    if 'ExterQual' in df.columns:
        df['ExterQual_ordinal'] = df['ExterQual'].map(quality_map).fillna(3)
    if 'BsmtQual' in df.columns:
        df['BsmtQual_ordinal'] = df['BsmtQual'].map(quality_map).fillna(0)
    if 'BsmtExposure' in df.columns:
        df['BsmtExposure_ordinal'] = df['BsmtExposure'].map(exposure_map).fillna(0)
    if 'GarageFinish' in df.columns:
        df['GarageFinish_ordinal'] = df['GarageFinish'].map(finish_map).fillna(0)
    
    # Add any missing required columns as 0
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0
    
    # Select only the required columns in the right order
    final_cols = []
    for f in feature_names:
        if f.startswith('num__'):
            col = f[5:]
        elif f.startswith('cat__'):
            col = f[5:]
        else:
            col = f
        if col in df.columns:
            final_cols.append(col)
    
    return df[final_cols]


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_kaggle_pipeline(use_existing_model: str = None):
    """Run the full Kaggle House Prices competition pipeline."""
    
    print_header("KAGGLE HOUSE PRICES COMPETITION")
    print("  Goal: Predict SalePrice for residential homes in Ames, Iowa")
    print("  Metric: RMSE on log(predicted) vs log(actual)")
    
    result = {}
    model_name = use_existing_model
    
    if use_existing_model:
        # Skip training, use existing model
        print_header("USING EXISTING MODEL")
        print(f"  Model: {use_existing_model}")
        model_name = use_existing_model
        
        # Verify model exists
        model_info = get_model_info(model_name)
        if not model_info:
            print(f"❌ Model '{model_name}' not found in registry!")
            return None
        
        print(f"  Type: {model_info.get('model_type')}")
        print(f"  Target: {model_info.get('target_column')}")
        print(f"  Features: {len(model_info.get('feature_names', []))}")
        
    else:
        # -------------------------------------------------------------------------
        # STEP 1: Load and Register Training Data
        # -------------------------------------------------------------------------
        print_header("STEP 1: Load and Register Training Data")
        
        if not TRAIN_FILE.exists():
            print(f"❌ Training file not found: {TRAIN_FILE}")
            return None
        
        df_train = pd.read_csv(TRAIN_FILE)
        print_step("LOADED", f"Training data from {TRAIN_FILE}")
        preview_dataframe(df_train, "Training Data")
        
        # Show target distribution
        print(f"\n  📈 Target Variable (SalePrice):")
        print(f"     Min: ${df_train['SalePrice'].min():,.0f}")
        print(f"     Max: ${df_train['SalePrice'].max():,.0f}")
        print(f"     Mean: ${df_train['SalePrice'].mean():,.0f}")
        print(f"     Median: ${df_train['SalePrice'].median():,.0f}")
        print(f"     Std: ${df_train['SalePrice'].std():,.0f}")
        
        # Log-transform target for better RMSE performance
        # The agent should handle this, but we can also do it here
        print_step("TRANSFORM", "Log-transforming SalePrice for RMSE optimization")
        df_train['SalePrice'] = np.log1p(df_train['SalePrice'])
        print(f"  Log(SalePrice) range: {df_train['SalePrice'].min():.2f} to {df_train['SalePrice'].max():.2f}")
        
        # Register the dataset
        register_dataset("house_prices_train", df_train)
        print_step("REGISTERED", "Dataset registered as 'house_prices_train'")
        
        # -------------------------------------------------------------------------
        # STEP 2: Invoke Training Agent
        # -------------------------------------------------------------------------
        print_header("STEP 2: Invoke Training Agent")
        
        print_step("GOAL", "")
        print(f"  {GOAL[:200]}...")
        print_step("MODEL", "No preference - LLM will select based on goal")
        print_step("INVOKE", "Calling invoke_training_agent()...")
        print("\n" + "-" * 80)
        
        result = invoke_training_agent(
            goal=GOAL,
            linked_datasets=["house_prices_train"],
            user_model_preference=None  # Let LLM choose!
        )
        
        print("-" * 80)
        
        # -------------------------------------------------------------------------
        # STEP 3: Review Results
        # -------------------------------------------------------------------------
        print_header("STEP 3: Training Results")
        
        print_step("SUMMARY", "")
        print(f"  Selected Model: {result.get('selected_model')}")
        print(f"  Model Explanation: {result.get('model_explanation', '')[:150]}...")
        print(f"  Collected Dataset: {result.get('collected_dataset_ref')}")
        print(f"  Cleaned Dataset: {result.get('cleaned_dataset_ref')}")
        print(f"  Train Dataset: {result.get('transformed_train_ref')}")
        print(f"  Model Weights: {result.get('model_weights_path')}")
        print(f"  Report Path: {result.get('report_path')}")
        
        training_metrics = result.get("training_metrics", {})
        if training_metrics:
            print_step("METRICS", "")
            print(f"  Success: {training_metrics.get('success')}")
            print(f"  Model Name: {training_metrics.get('model_name')}")
            print(f"  Iterations: {training_metrics.get('num_iterations', 0)}")
            
            # For regression, look for R² or RMSE
            val_acc = training_metrics.get('val_accuracy')
            if val_acc:
                print(f"  Val Score: {val_acc}")
            test_acc = training_metrics.get('test_accuracy')
            if test_acc:
                print(f"  Test Score: {test_acc}")
        
        # Check for errors
        if result.get("error"):
            print(f"\n❌ Error: {result.get('error')}")
            return result
        
        model_name = result.get("model_weights_path")
        if not model_name:
            print("\n❌ No model was trained!")
            return result
    
    # -------------------------------------------------------------------------
    # STEP 4: Load Test Data and Apply Feature Engineering
    # -------------------------------------------------------------------------
    print_header("STEP 4: Prepare Test Data")
    
    if not TEST_FILE.exists():
        print(f"❌ Test file not found: {TEST_FILE}")
        return result
    
    df_test_raw = pd.read_csv(TEST_FILE)
    print_step("LOADED", f"Test data from {TEST_FILE}")
    preview_dataframe(df_test_raw, "Test Data (Raw)")
    
    # Save the Id column for submission
    test_ids = df_test_raw["Id"].copy()
    
    # Load the trained model
    print_step("LOAD MODEL", f"Loading model '{model_name}'")
    
    try:
        model = load_model(model_name)
        model_info = get_model_info(model_name)
        print(f"  Model type: {model_info.get('model_type')}")
        print(f"  Features: {len(model_info.get('feature_names', []))} features")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return result
    
    # Get the feature names the model expects
    feature_names = model_info.get("feature_names", [])
    print(f"  Model expects features: {feature_names[:10]}...")
    
    # Apply the same feature engineering to test data
    print_step("FEATURE ENGINEERING", "Applying transformations to test data")
    
    df_test = df_test_raw.drop(columns=["Id"], errors="ignore").copy()
    
    # Apply the same feature engineering that was applied to training data
    # The function will strip 'num__' prefix and create matching columns
    X_test = apply_feature_engineering(df_test, feature_names)
    
    print(f"  Final test shape: {X_test.shape}")
    
    # -------------------------------------------------------------------------
    # STEP 5: Generate Predictions
    # -------------------------------------------------------------------------
    print_header("STEP 5: Generate Predictions")
    
    print_step("PREDICT", "Generating predictions...")
    
    try:
        predictions = model.predict(X_test)
        print(f"  Generated {len(predictions)} predictions")
        
        # Reverse log transform (we log-transformed the target)
        predictions_dollars = np.expm1(predictions)
        print(f"  Price range: ${predictions_dollars.min():,.0f} to ${predictions_dollars.max():,.0f}")
        print(f"  Mean predicted price: ${predictions_dollars.mean():,.0f}")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return result
    
    # -------------------------------------------------------------------------
    # STEP 6: Create Submission File
    # -------------------------------------------------------------------------
    print_header("STEP 6: Create Submission File")
    
    submission = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": predictions_dollars
    })
    
    submission.to_csv(SUBMISSION_FILE, index=False)
    print_step("SAVED", f"Submission saved to {SUBMISSION_FILE}")
    
    preview_dataframe(submission, "Submission Preview", max_rows=5)
    
    print(f"\n  📁 File: {SUBMISSION_FILE}")
    print(f"  📊 Rows: {len(submission)}")
    print(f"  ✅ Ready to upload to Kaggle!")
    
    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------
    print_header("PIPELINE COMPLETE")
    
    print(f"""
  📋 Summary:
     - Test samples: {len(df_test_raw)}
     - Model: {model_name}
     - Submission: {SUBMISSION_FILE}
  
  📤 Next Steps:
     1. Go to https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/submit
     2. Upload {SUBMISSION_FILE.name}
     3. Check your score!
  
  💡 Tips for improvement:
     - Try different models (run with user_model_preference="xgboost" or "random_forest")
     - Feature engineering: add more interaction terms
     - Handle missing values more carefully
     - Try ensemble methods
""")
    
    return result


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Kaggle House Prices competition pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Only load data, don't train")
    parser.add_argument("--model", type=str, help="Use existing model instead of training new")
    args = parser.parse_args()
    
    if args.dry_run:
        print_header("DRY RUN - Data Loading Only")
        
        df_train = pd.read_csv(TRAIN_FILE)
        df_test = pd.read_csv(TEST_FILE)
        
        preview_dataframe(df_train, "Training Data")
        preview_dataframe(df_test, "Test Data")
        
        print(f"\n  SalePrice stats:")
        print(df_train["SalePrice"].describe())
    else:
        result = run_kaggle_pipeline(use_existing_model=args.model)
        
        if result is None:
            print("\n❌ Pipeline failed!")
        elif isinstance(result, dict) and result.get("training_metrics", {}).get("success"):
            print("\n✅ Pipeline completed successfully!")
        else:
            print("\n⚠️ Pipeline completed - check output above")
